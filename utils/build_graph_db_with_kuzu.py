"""
HybridKuzuRAG: Graph(구조) + Vector(문맥) 하이브리드 검색 시스템
- Retriever는 Documents 객체만 반환 (LLM 답변 생성 없음)
"""

from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_kuzu import KuzuGraph, KuzuQAChain
from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
import kuzu
from typing import List, Optional, Set, Tuple, Dict
from collections import defaultdict


class HybridKuzuRAG:
    """
    그래프 구조 + 문맥 검색을 결합한 Hybrid RAG 시스템
    
    특징:
    - 자동 노드/관계 추출 (LLM 기반)
    - 그래프 구조 검색 (엔티티 관계)
    - 벡터 유사도 검색 (문맥/의미)
    - Documents 객체 직접 반환 (LLM 답변 생성 없음)
    """
    
    def __init__(
        self, 
        llm_model: str = "gpt-4o",
        embedding_model: str = "text-embedding-3-small",
        in_memory: bool = True,
        db_path: Optional[str] = None
    ):
        """
        초기화
        
        Args:
            llm_model: LLM 모델 이름 (그래프 추출용)
            embedding_model: Embedding 모델 이름
            in_memory: True면 메모리에, False면 디스크에 저장
            db_path: 디스크 저장 시 경로 (in_memory=False일 때 필수)
        """
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        
        # Kuzu GraphDB 초기화
        if in_memory or db_path is None:
            print("🧠 In-Memory 모드로 Kuzu 데이터베이스 생성")
            self.db = kuzu.Database()
            self.db_path = ":memory:"
        else:
            print(f"💾 디스크 기반 모드로 Kuzu 데이터베이스 생성: {db_path}")
            self.db = kuzu.Database(db_path)
            self.db_path = db_path
        
        self.conn = kuzu.Connection(self.db)
        
        # Vector Store
        self.vector_store = None
        self.original_documents = []
        
        # 스키마 정보 저장
        self.discovered_node_types: Set[str] = set()
        self.discovered_relationships: Set[Tuple[str, str, str]] = set()
        self.node_properties: defaultdict = defaultdict(set)
        self.rel_properties: defaultdict = defaultdict(set)
        
        # Graph Transformer 초기화 (스키마 제약 없음 - 자동 추출)
        self.graph_transformer = LLMGraphTransformer(
            llm=self.llm,
            node_properties=True,
            relationship_properties=True
        )
        
        # Graph 래퍼
        self.graph = None
        self.graph_chain = None
    
    def build_from_documents(self, documents: List[Document]) -> List:
        """
        Document 리스트로부터 Hybrid RAG 시스템 구축
        
        Args:
            documents: LangChain Document 객체 리스트
            
        Returns:
            graph_documents: 추출된 그래프 문서 리스트
        """
        print("=" * 70)
        print("🔧 HYBRID RAG 구축 시작")
        print("=" * 70)
        print("  전략: Graph (엔티티/관계 구조) + Vector (문맥/의미)")
        print()
        
        self.original_documents = documents
        
        # ========================================
        # STEP 1: 그래프 구조 추출 (LLM 자동)
        # ========================================
        print("📊 [STEP 1/3] 그래프 구조 추출 중...")
        print("  - LLM이 문서에서 엔티티와 관계를 자동 추출합니다")
        
        graph_documents = self.graph_transformer.convert_to_graph_documents(documents)
        
        total_nodes = sum(len(gd.nodes) for gd in graph_documents)
        total_rels = sum(len(gd.relationships) for gd in graph_documents)
        
        print(f"  ✓ {total_nodes}개 노드 추출 완료")
        print(f"  ✓ {total_rels}개 관계 추출 완료")
        
        # 스키마 학습
        print("\n  📚 추출된 엔티티 타입:")
        for graph_doc in graph_documents:
            for node in graph_doc.nodes:
                if node.type not in self.discovered_node_types:
                    print(f"    - {node.type}")
                self.discovered_node_types.add(node.type)
                
                for prop in node.properties.keys():
                    self.node_properties[node.type].add(prop)
            
            for rel in graph_doc.relationships:
                rel_tuple = (rel.source.type, rel.type, rel.target.type)
                self.discovered_relationships.add(rel_tuple)
                
                for prop in rel.properties.keys():
                    self.rel_properties[rel.type].add(prop)
        
        print(f"\n  ✓ 총 {len(self.discovered_node_types)}개 노드 타입")
        print(f"  ✓ 총 {len(self.discovered_relationships)}개 관계 타입")
        
        # Kuzu 스키마 생성 및 데이터 삽입
        print("\n  🏗️  Kuzu 스키마 생성 중...")
        self._create_kuzu_schema()
        
        print("  💾 그래프 데이터 삽입 중...")
        inserted_count = self._insert_graph_data(graph_documents)
        print(f"  ✓ {inserted_count}개 노드 삽입 완료")
        
        # ========================================
        # STEP 2: 벡터 스토어 구축
        # ========================================
        print("\n🔢 [STEP 2/3] 벡터 스토어 구축 중...")
        print("  - 문서 전체를 임베딩하여 의미 검색을 지원합니다")
        
        self.vector_store = FAISS.from_documents(
            documents,
            self.embeddings
        )
        print(f"  ✓ {len(documents)}개 문서 임베딩 완료")
        
        # ========================================
        # STEP 3: Graph 래퍼 초기화
        # ========================================
        self.graph = KuzuGraph(self.db, allow_dangerous_requests=True)
        self.graph_chain = KuzuQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            verbose=False,
            allow_dangerous_requests=True
        )
        
        # ========================================
        # STEP 4: 완료
        # ========================================
        print("\n✅ [STEP 3/3] Hybrid RAG 구축 완료!")
        print("=" * 70)
        print("  📊 그래프: 엔티티 관계 구조 (누가, 무엇을, 어떻게)")
        print("  🔢 벡터: 문맥 의미 검색 (유사한 내용 찾기)")
        print("=" * 70)
        
        return graph_documents
    
    def _create_kuzu_schema(self):
        """학습된 스키마로 Kuzu 테이블 동적 생성"""
        
        # 노드 테이블 생성
        for node_type in self.discovered_node_types:
            properties = self.node_properties[node_type]
            
            # 기본 구조: id + PRIMARY KEY
            prop_definitions = ["id STRING", "PRIMARY KEY(id)"]
            
            # 추가 속성
            for prop in properties:
                if prop != 'id':
                    prop_definitions.append(f"{prop} STRING")
            
            create_query = f"""
                CREATE NODE TABLE IF NOT EXISTS {node_type} (
                    {', '.join(prop_definitions)}
                )
            """
            
            try:
                self.conn.execute(create_query)
            except Exception as e:
                print(f"    ⚠️  {node_type} 테이블 생성 실패: {e}")
        
        # 관계 테이블 생성
        for source_type, rel_type, target_type in self.discovered_relationships:
            properties = self.rel_properties[rel_type]
            
            # 관계 이름 정규화 (특수문자 제거)
            rel_name = f"{source_type}_{rel_type}_{target_type}".replace("-", "_").replace(" ", "_")
            
            prop_definitions = []
            for prop in properties:
                prop_definitions.append(f"{prop} STRING")
            
            if prop_definitions:
                create_query = f"""
                    CREATE REL TABLE IF NOT EXISTS {rel_name} (
                        FROM {source_type} TO {target_type},
                        {', '.join(prop_definitions)}
                    )
                """
            else:
                create_query = f"""
                    CREATE REL TABLE IF NOT EXISTS {rel_name} (
                        FROM {source_type} TO {target_type}
                    )
                """
            
            try:
                self.conn.execute(create_query)
            except Exception as e:
                print(f"    ⚠️  {rel_name} 관계 생성 실패: {e}")
    
    def _insert_graph_data(self, graph_documents) -> int:
        """
        그래프 데이터를 Kuzu에 삽입
        
        Returns:
            inserted_count: 삽입된 노드 개수
        """
        inserted_nodes = set()
        
        for graph_doc in graph_documents:
            # 노드 삽입
            for node in graph_doc.nodes:
                node_key = (node.type, node.id)
                if node_key in inserted_nodes:
                    continue
                
                # 속성 처리 (특수문자 이스케이프)
                props = {"id": node.id}
                for k, v in node.properties.items():
                    if k != 'id':
                        # SQL injection 방지: 작은따옴표 이스케이프
                        props[k] = str(v).replace("'", "''")
                
                # INSERT 쿼리 생성
                columns = ', '.join(props.keys())
                values = ', '.join([f"'{v}'" for v in props.values()])
                
                insert_query = f"""
                    CREATE (:{node.type} {{{columns}: [{values}]}})
                """
                
                try:
                    self.conn.execute(insert_query)
                    inserted_nodes.add(node_key)
                except Exception as e:
                    # 중복 등의 에러는 무시
                    pass
            
            # 관계 삽입
            for rel in graph_doc.relationships:
                rel_name = f"{rel.source.type}_{rel.type}_{rel.target.type}".replace("-", "_").replace(" ", "_")
                
                # 관계 속성 처리
                if rel.properties:
                    props_str = ', '.join([
                        f"{k}: '{str(v).replace(chr(39), chr(39)+chr(39))}'" 
                        for k, v in rel.properties.items()
                    ])
                    match_query = f"""
                        MATCH (a:{rel.source.type}), (b:{rel.target.type})
                        WHERE a.id = '{rel.source.id}' AND b.id = '{rel.target.id}'
                        CREATE (a)-[:{rel_name} {{{props_str}}}]->(b)
                    """
                else:
                    match_query = f"""
                        MATCH (a:{rel.source.type}), (b:{rel.target.type})
                        WHERE a.id = '{rel.source.id}' AND b.id = '{rel.target.id}'
                        CREATE (a)-[:{rel_name}]->(b)
                    """
                
                try:
                    self.conn.execute(match_query)
                except Exception as e:
                    # 중복 관계 등의 에러는 무시
                    pass
        
        return len(inserted_nodes)
    
    def create_retriever(
        self, 
        search_mode: str = "both",
        vector_k: int = 5
    ):
        """
        하이브리드 Retriever 생성 (Documents 객체 반환)
        
        Args:
            search_mode: 검색 모드
                - "graph": 그래프 구조만 사용 (엔티티/관계)
                - "vector": 벡터 검색만 사용 (문맥/의미)
                - "both": 하이브리드 (둘 다 사용) ⭐ 권장
            vector_k: 벡터 검색 시 반환할 문서 개수
            
        Returns:
            HybridRetriever: Documents를 반환하는 커스텀 Retriever
        """
        
        print("\n" + "=" * 70)
        print("🔍 Hybrid Retriever 생성")
        print("=" * 70)
        print(f"  모드: {search_mode.upper()}")
        print(f"  반환 타입: List[Document]")
        
        return HybridRetriever(
            hybrid_rag=self,
            search_mode=search_mode,
            vector_k=vector_k
        )
    
    def get_schema_info(self):
        """학습된 그래프 스키마 정보 출력"""
        
        print("\n" + "=" * 70)
        print("📊 그래프 스키마 정보")
        print("=" * 70)
        
        if not self.discovered_node_types:
            print("  ⚠️  아직 그래프가 구축되지 않았습니다.")
            return
        
        print("\n【 노드 타입 】")
        for node_type in sorted(self.discovered_node_types):
            props = self.node_properties[node_type]
            print(f"\n  📌 {node_type}")
            if props:
                print(f"     속성: {', '.join(sorted(props))}")
        
        print("\n【 관계 타입 】")
        for source, rel, target in sorted(self.discovered_relationships):
            props = self.rel_properties[rel]
            print(f"\n  🔗 ({source}) -[{rel}]-> ({target})")
            if props:
                print(f"     속성: {', '.join(sorted(props))}")
        
        print("\n" + "=" * 70)
    
    def query_graph_directly(self, cypher_query: str):
        """
        Cypher 쿼리 직접 실행 (디버깅용)
        
        Args:
            cypher_query: Cypher 쿼리 문자열
            
        Returns:
            결과 리스트
        """
        try:
            result = self.conn.execute(cypher_query)
            return result.get_as_pl()  # Polars DataFrame으로 반환
        except Exception as e:
            print(f"쿼리 실행 실패: {e}")
            return None
    
    def get_statistics(self) -> Dict:
        """
        시스템 통계 정보 반환
        
        Returns:
            통계 딕셔너리
        """
        stats = {
            "node_types": len(self.discovered_node_types),
            "relationship_types": len(self.discovered_relationships),
            "documents": len(self.original_documents),
            "mode": "in-memory" if self.db_path == ":memory:" else "disk-based",
            "db_path": self.db_path
        }
        
        return stats
    
    def print_statistics(self):
        """시스템 통계 정보 출력"""
        
        stats = self.get_statistics()
        
        print("\n" + "=" * 70)
        print("📈 시스템 통계")
        print("=" * 70)
        print(f"  노드 타입: {stats['node_types']}개")
        print(f"  관계 타입: {stats['relationship_types']}개")
        print(f"  원본 문서: {stats['documents']}개")
        print(f"  저장 모드: {stats['mode']}")
        print(f"  DB 경로: {stats['db_path']}")
        print("=" * 70)


class HybridRetriever(BaseRetriever):
    """
    Documents 객체를 직접 반환하는 커스텀 Retriever
    LLM 답변 생성 없이 검색된 문서만 반환
    """
    
    hybrid_rag: HybridKuzuRAG
    search_mode: str = "both"
    vector_k: int = 5
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        검색 실행 및 Documents 반환
        
        Args:
            query: 검색 질의
            
        Returns:
            검색된 Document 객체 리스트
        """
        documents = []
        
        # ========================================
        # 1. 그래프 검색
        # ========================================
        if self.search_mode in ["graph", "both"]:
            try:
                # Cypher 쿼리 생성 및 실행
                graph_result = self.hybrid_rag.graph_chain.invoke(query)
                
                # 그래프 검색 결과를 Document로 변환
                graph_content = graph_result.get('result', '')
                
                if graph_content and graph_content != '':
                    graph_doc = Document(
                        page_content=graph_content,
                        metadata={
                            "source": "graph_search",
                            "search_type": "graph",
                            "query": query
                        }
                    )
                    documents.append(graph_doc)
            
            except Exception as e:
                print(f"  ⚠️  그래프 검색 실패: {e}")
        
        # ========================================
        # 2. 벡터 검색
        # ========================================
        if self.search_mode in ["vector", "both"]:
            try:
                # 벡터 유사도 검색
                vector_docs = self.hybrid_rag.vector_store.similarity_search(
                    query, 
                    k=self.vector_k
                )
                
                # 메타데이터에 검색 타입 추가
                for doc in vector_docs:
                    doc.metadata["search_type"] = "vector"
                    doc.metadata["query"] = query
                
                documents.extend(vector_docs)
            
            except Exception as e:
                print(f"  ⚠️  벡터 검색 실패: {e}")
        
        return documents
    
    async def _aget_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """비동기 검색 (동기 버전 호출)"""
        return self._get_relevant_documents(query, run_manager=run_manager)