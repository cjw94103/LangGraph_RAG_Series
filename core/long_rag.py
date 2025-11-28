from utils.llm_chain_factory import make_llm_chain, make_structed_llm_chain

from typing import Annotated, TypedDict, List, Literal, Any, Dict
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_teddynote.messages import messages_to_history
from utils.helper import extract_model_provider, model_name_to_max_tokens, load_string_from_txt
from langchain_core.runnables import RunnableConfig

from langchain.schema import Document
from collections import defaultdict
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field

# 필요 함수
def map_chunks_to_original_pages(
    retrieved_chunks: List[Document],
    vectorstore: FAISS,
    source_key: str = "source",
    page_key: str = "page",
    chunk_index_key: str = "chunk_index"
) -> List[Dict[str, Any]]:
    """
    검색된 청크들이 속한 원본 문서의 해당 페이지 전체 내용을 재구성합니다.
    검색된 청크뿐만 아니라 같은 문서의 같은 페이지에 있는 모든 청크를 가져와서 원본 페이지를 복원합니다.
    
    Args:
        retrieved_chunks: 검색된 Document 청크 리스트
        vectorstore: 전체 문서가 저장된 FAISS vectorstore
        source_key: 원본 문서를 식별하는 메타데이터 키 (기본값: "source")
        page_key: 페이지를 식별하는 메타데이터 키 (기본값: "page")
        chunk_index_key: 청크 인덱스 메타데이터 키 (기본값: "chunk_index")
    
    Returns:
        문서/페이지별로 재구성된 원본 페이지 전체 텍스트 리스트
        [
            {
                "source": str,           # 문서명
                "page": int,             # 페이지 번호
                "content": str,          # 원본 페이지 전체 텍스트
                "chunk_indices": [int],  # 페이지의 모든 청크 인덱스들
                "chunk_count": int,      # 페이지의 전체 청크 수
                "retrieved_chunk_indices": [int]  # 실제 검색된 청크 인덱스들
            },
            ...
        ]
    """
    # 1. 검색된 청크들의 (문서, 페이지) 조합 추출
    retrieved_pages = set()
    retrieved_chunk_map = defaultdict(list)
    
    for chunk in retrieved_chunks:
        source = chunk.metadata.get(source_key, "unknown")
        page = chunk.metadata.get(page_key, 0)
        chunk_idx = chunk.metadata.get(chunk_index_key, 0)
        
        page_key_tuple = (source, page)
        retrieved_pages.add(page_key_tuple)
        retrieved_chunk_map[page_key_tuple].append(chunk_idx)
    
    # 2. vectorstore에서 모든 문서 가져오기
    all_docs = list(vectorstore.docstore._dict.values())
    
    # 3. 각 (문서, 페이지)에 대해 해당 페이지의 모든 청크 수집
    reconstructed_pages = []
    
    for (source, page) in retrieved_pages:
        # 같은 문서, 같은 페이지의 모든 청크 찾기
        page_chunks = [
            doc for doc in all_docs
            if doc.metadata.get(source_key) == source 
            and doc.metadata.get(page_key) == page
        ]
        
        if not page_chunks:
            continue
        
        # 청크 인덱스로 정렬
        page_chunks.sort(key=lambda x: x.metadata.get(chunk_index_key, 0))
        
        # 원본 페이지 전체 텍스트 재구성
        original_page_content = "\n\n".join([chunk.page_content for chunk in page_chunks])
        all_chunk_indices = [chunk.metadata.get(chunk_index_key, 0) for chunk in page_chunks]
        
        reconstructed_pages.append({
            "source": source,
            "page": page,
            "content": original_page_content,
            "chunk_indices": sorted(all_chunk_indices),
            "chunk_count": len(page_chunks),
            "retrieved_chunk_indices": sorted(retrieved_chunk_map[(source, page)]),
            "metadata": page_chunks[0].metadata if page_chunks else {}
        })
    
    # 4. 결과를 문서명과 페이지 순으로 정렬
    reconstructed_pages.sort(key=lambda x: (x["source"], x["page"]))
    
    return reconstructed_pages

## content template
content_template = """source : {source}
page : {page}
content " {content}
------------------------------------delimeter------------------------------------"""

## graph class
class GraphState(TypedDict):
    question : str
    answer : str

    result_chunks : List[Document]
    Ig : str
    cot_result : str
    Id : str

    chat_history : Annotated[list, add_messages]

    llm_sys_prompt : str

class GraphConfig(TypedDict, total=False):
    answer_llm_name : str
    answer_llm_temperature : float

class RelevanceQuery(BaseModel):
    """A system that determines whether retrieved chunks are relevant to the user's question."""
    
    is_relevant: Literal["True", "False"] = Field(
        ...,
        description="Determine whether the retrieved chunk content is relevant to the user's question and respond with either 'True' or 'False'.",
    )

class LongRAG:
    def __init__(self, dense_retriever, top_k=5):
        self.dense_retriever = dense_retriever
        self.dense_retriever.search_kwargs["k"] = top_k
        
        self.llm_ie = make_llm_chain(system_prompt_or_path="",
                                   human_message=load_string_from_txt("./sys_prompt_hub/02_long_rag/01_llm_ie_human_message.txt"),
                                   temperature=0,
                                   top_p=None,
                                   max_tokens=model_name_to_max_tokens("gpt-4.1-nano"),
                                   default_model="gpt-4.1-nano",
                                   default_provider=extract_model_provider("gpt-4.1-nano"),
                                   configurable=True)

        self.llm_cot_guide = make_llm_chain(system_prompt_or_path="",
                                           human_message=load_string_from_txt("./sys_prompt_hub/02_long_rag/02_llm_cot_guide_human_message.txt"),
                                           temperature=0,
                                           top_p=None,
                                           max_tokens=model_name_to_max_tokens("gpt-4.1-nano"),
                                           default_model="gpt-4.1-nano",
                                           default_provider=extract_model_provider("gpt-4.1-nano"),
                                           configurable=True)

        self.llm_filter = make_structed_llm_chain(system_prompt_or_path="",
                                                 human_message=load_string_from_txt("./sys_prompt_hub/02_long_rag/03_llm_filter_human_message.txt"),
                                                 temperature=0,
                                                 top_p=None,
                                                 max_tokens=model_name_to_max_tokens("gpt-4.1-nano"),
                                                 default_model="gpt-4.1-nano",
                                                 default_provider=extract_model_provider("gpt-4.1-nano"),
                                                 structed_class=RelevanceQuery,
                                                 configurable=True)

        self.answer_llm = make_llm_chain(system_prompt_or_path="{llm_sys_prompt}",
                                        human_message=load_string_from_txt("./sys_prompt_hub/02_long_rag/04_01_llm_answer_human_message.txt"),
                                        temperature=0,
                                        top_p=None,
                                        max_tokens=model_name_to_max_tokens("gpt-4.1-mini"),
                                        default_model="gpt-4.1-mini",
                                        default_provider=extract_model_provider("gpt-4.1-mini"),
                                        configurable=True)

        self.memory = MemorySaver()

    def retrieve_documents(self, state : GraphState, config : RunnableConfig):
        question = state["question"]
        retrieve_results = self.dense_retriever.invoke(question)

        return {"result_chunks" : retrieve_results}

    def extract_Ig(self, state : GraphState, config : RunnableConfig):
        question = state["question"]
        result_chunks = state["result_chunks"]

        ori_para = map_chunks_to_original_pages(result_chunks, self.dense_retriever.vectorstore)
        content = ""

        for para in ori_para:
            content += content_template.format(source=para["source"], page=para["page"], content=para["content"])

        Ig = self.llm_ie.invoke({"content" : content, "question" : question})

        return {"Ig" : Ig}

    def cot_guidance(self, state : GraphState, config : RunnableConfig):
        question = state["question"]
        result_chunks = state["result_chunks"]

        content_chunks = ""
        for chunk in result_chunks:
            content_chunks += content_template.format(source=chunk.metadata["source"], page=str(chunk.metadata["page"]), content=chunk.page_content)

        cot_result = self.llm_cot_guide.invoke({"content" : content_chunks, "question" : question})

        return {"cot_result" : cot_result}

    def extract_Id(self, state : GraphState, config : RunnableConfig):
        question = state["question"]
        result_chunks = state["result_chunks"]
        cot_result = state["cot_result"]

        filter_inputs = [
            {
                "content": content_template.format(
                    source="", 
                    page="", 
                    content=chunk.page_content
                ).replace("------------------------------------delimeter------------------------------------", ""), 
                "question": question, 
                "CoT": cot_result
            } 
            for chunk in result_chunks
        ]

        filtering_results = self.llm_filter.batch(filter_inputs, config={"max_concurrency": 4})

        include_idx = [i for i, x in enumerate(filtering_results) if x == "True"]
        filtering_documents = [result_chunks[include_idx[i]] for i in range(len(include_idx))]
        
        Id = ""
        for chunk in filtering_documents:
            Id += content_template.format(source=chunk.metadata["source"], page=str(chunk.metadata["page"]), content=chunk.page_content)

        return {"Id" : Id}

    def llm_answer(self, state : GraphState, config : RunnableConfig):
        question = state["question"]
        Id = state["Id"]
        Ig = state["Ig"]
        chat_history = messages_to_history(state["chat_history"])
        
        llm_sys_prompt = state["llm_sys_prompt"]
        answer_llm_name = config.get("configurable", {}).get("answer_llm_name")
        answer_llm_temperature = config.get("configurable", {}).get("answer_llm_temperature")

        # streaming 대상만 이 코드를 사용
        chain = self.answer_llm.with_config(configurable={"model": answer_llm_name, 
                                                         "model_provider": extract_model_provider(answer_llm_name),
                                                         "max_tokens" : model_name_to_max_tokens(answer_llm_name),
                                                         "temperature" : answer_llm_temperature})
        

        answer = chain.invoke({"llm_sys_prompt": llm_sys_prompt, "Ig" : Ig, "Id" : Id, "question" : question, "chat_history" : chat_history})

        return {"answer" : answer, "chat_history" : [("user", question), ("assistant", answer)]}

    def make_app_graph(self):
        builder = StateGraph(GraphState, config_schema=GraphConfig)

        # 노드 추가
        builder.add_node("retrieve_documents", self.retrieve_documents)
        builder.add_node("extract_Ig", self.extract_Ig)
        builder.add_node("cot_guidance", self.cot_guidance)
        builder.add_node("extract_Id", self.extract_Id)
        builder.add_node("llm_answer", self.llm_answer)

        # 엣지 연결
        builder.add_edge(START, "retrieve_documents")
        builder.add_edge("retrieve_documents", "extract_Ig")
        builder.add_edge("extract_Ig", "cot_guidance")
        builder.add_edge("cot_guidance", "extract_Id")
        builder.add_edge("extract_Id", "llm_answer")

        builder.add_edge("llm_answer", END)

        graph = builder.compile(checkpointer=self.memory)

        return graph