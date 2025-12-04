import numpy as np

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

## content template
content_template = """source : {source}
page : {page}
content " {content}
------------------------------------delimeter------------------------------------"""

class DocumentScoring(BaseModel):
    score: float = Field(description="You must output a relevance score as a float between -1.0 and 1.0")

class GraphConfig(TypedDict, total=False):
    answer_llm_name : str
    answer_llm_temperature : float
    
class GraphState(TypedDict):
    question : str
    answer : str
    
    # document retrieve
    result_chunks : List[Document]
    doc_q_a_triplet : List[dict]

    chat_history : Annotated[list, add_messages]

class MAINRAG:
    def __init__(self, retriever, top_k=5, doc_score_n=0.5):
        self.retriever = retriever
        self.retriever.search_kwargs["k"] = top_k
        self.doc_score_n = doc_score_n

        self.predictor_llm = make_llm_chain(system_prompt_or_path=load_string_from_txt("./sys_prompt_hub/06_main_rag/01_predictor_prompt.txt"),
                                            human_message="Documents : {documents}\n\nQuestion : {question}",
                                            temperature=0,
                                            top_p=None,
                                            max_tokens=model_name_to_max_tokens("gpt-4.1-nano"),
                                            default_model="gpt-4.1-nano",
                                            default_provider=extract_model_provider("gpt-4.1-nano"),
                                            configurable=True)

        self.document_score_llm = make_structed_llm_chain(system_prompt_or_path=load_string_from_txt("./sys_prompt_hub/06_main_rag/02_document_score_prmpt.txt"),
                                                          human_message="Document : {document}\n\nUser's question : {question}\n\nAnswer : {answer}",
                                                          temperature=0,
                                                          top_p=None,
                                                          max_tokens=model_name_to_max_tokens("gpt-4.1-nano"),
                                                          default_model="gpt-4.1-nano",
                                                          default_provider=extract_model_provider("gpt-4.1-nano"),
                                                          structed_class=DocumentScoring,
                                                          configurable=True)

        self.answer_llm = make_llm_chain(system_prompt_or_path="./sys_prompt_hub/06_main_rag/03_answer_llm_prompt.txt",
                                        human_message="user's question:\n{question}\n\nreference documents:\n{documents}",
                                        temperature=0,
                                        top_p=None,
                                        max_tokens=model_name_to_max_tokens("gpt-4.1-mini"),
                                        default_model="gpt-4.1-mini",
                                        default_provider=extract_model_provider("gpt-4.1-mini"),
                                        configurable=True)

        self.memory = MemorySaver()

    def retrieve_documents(self, state : GraphState, config : RunnableConfig):
        question = state["question"]
        result_chunks = self.retriever.invoke(question)

        return {"result_chunks" : result_chunks}

    def doc_qa_predict(self, state : GraphState, config : RunnableConfig):
        question = state["question"]
        result_chunks = state["result_chunks"]

        predict_inputs = [
            {
                "documents": content_template.format(
                    source="", 
                    page="", 
                    content=chunk.page_content
                ).replace("------------------------------------delimeter------------------------------------", ""), 
                "question": question
            } 
            for chunk in result_chunks
        ]

        predict_result = self.predictor_llm.batch(predict_inputs, config={"max_concurrency": 4})

        doc_q_a_list = []
        for inputs, result in zip(predict_inputs, predict_result):
            inputs["answer"] = result
            doc_q_a_list.append(inputs)

        return {"doc_q_a_triplet" : doc_q_a_list}

    def document_scoring(self, state : GraphState, config : RunnableConfig):
        doc_q_a_triplet = state["doc_q_a_triplet"]
        scoring_inputs = [{"document": triplet["documents"], 
                           "question": triplet["question"], 
                           "answer": triplet["answer"]} for triplet in doc_q_a_triplet]
        
        scoring_results = self.document_score_llm.batch(scoring_inputs, config={"max_concurrency": 4})
        scoring_results = [result.score for result in scoring_results]  # 리스트로 유지
        
        # adaptive judge bar
        scores_array = np.array(scoring_results)  # 계산을 위해 임시로 NumPy 사용
        stan_dev = scores_array.std()
        tau_q = scores_array.mean()
        confidence_thresh = tau_q - (self.doc_score_n * stan_dev)
        
        # 필터링 후 점수 높은 순으로 sorting (reranking)
        # NumPy 없이 순수 Python으로 처리
        scored_items = [
            (i, score) for i, score in enumerate(scoring_results) 
            if score > confidence_thresh
        ]
        
        # 점수 기준 내림차순 정렬
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        # 필터링 및 정렬된 triplet 생성
        target_triplet_list = []
        for idx, score in scored_items:
            target_triplet = doc_q_a_triplet[idx].copy()  # 원본 보존을 위해 copy
            target_triplet["score"] = float(score)  # float로 명시적 변환
            target_triplet_list.append(target_triplet)
        
        # 수정: scoring_results가 아닌 target_triplet_list를 반환
        return {"doc_q_a_triplet": target_triplet_list}

    def llm_answer(self, state : GraphState, config : RunnableConfig):
        question = state["question"]
        doc_q_a_triplet = state["doc_q_a_triplet"]
        documents = ""

        for triplet in doc_q_a_triplet:
            doc = triplet["documents"]
            documents += doc + "\n\n---------------------delimeter---------------------"

        answer_llm_name = config.get("configurable", {}).get("answer_llm_name")
        answer_llm_temperature = config.get("configurable", {}).get("answer_llm_temperature")

        chain = self.answer_llm.with_config(configurable={"model": answer_llm_name, 
                                                         "model_provider": extract_model_provider(answer_llm_name),
                                                         "max_tokens" : model_name_to_max_tokens(answer_llm_name),
                                                         "temperature" : answer_llm_temperature})
        
        answer = chain.invoke({"question" : question, "documents" : documents})

        return {"answer" : answer, "chat_history" : [("user", question), ("assistant", answer)]}

    
    def make_app_graph(self):
        builder = StateGraph(GraphState, config_schema=GraphConfig)

        # 노드 추가
        builder.add_node("retrieve_documents", self.retrieve_documents)
        builder.add_node("doc_qa_predict", self.doc_qa_predict)
        builder.add_node("document_scoring", self.document_scoring)
        builder.add_node("llm_answer", self.llm_answer)

        # 엣지 연결
        builder.add_edge(START, "retrieve_documents")
        builder.add_edge("retrieve_documents", "doc_qa_predict")
        builder.add_edge("doc_qa_predict", "document_scoring")
        builder.add_edge("document_scoring", "llm_answer")
        builder.add_edge("llm_answer", END)

        graph = builder.compile(checkpointer=self.memory)

        return graph