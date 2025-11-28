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

class GraphState(TypedDict):
    question : str
    answer : str
    result_chunks : List[Document]
    chat_history : Annotated[list, add_messages]
    llm_sys_prompt : str

class GraphConfig(TypedDict, total=False):
    answer_llm_name : str
    answer_llm_temperature : float

class GradeDocuments(BaseModel):
    """A binary score to determine the relevance of the retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class Groundednesss(BaseModel):
    """A binary score indicating whether the generated answer is grounded in the facts."""

    # 답변이 사실에 기반하고 있는지 여부를 'yes' 또는 'no'로 나타내는 필드
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class GradeAnswer(BaseModel):
    """A binary score indicating whether the question is addressed."""

    # 답변의 관련성 평가: 'yes' 또는 'no'로 표기(yes: 관련성 있음, no: 관련성 없음)
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

class QueryRewrite(BaseModel):
    """A system that rewrites user questions to be clearer and more effective for web search."""
    
    rewritten_query: str = Field(
        ...,
        description=(
            "The rewritten version of the user's question that is:\n"
            "- Clear and specific with no ambiguous pronouns\n"
            "- Optimized for web search with searchable keywords\n"
            "- Self-contained with necessary context\n"
            "- Concise but complete\n"
            "- In the same language as the original question"
        ),
    )

class SELFRAG:
    def __init__(self, retriever, top_k=5):
        self.retriever = retriever
        self.retriever.search_kwargs["k"] = top_k

        self.grade_document_llm = make_structed_llm_chain(system_prompt_or_path=load_string_from_txt("./sys_prompt_hub/04_self_rag/01_grade_documents.txt"),
                                                          human_message="user's question : {question}\n\nretrieved documents : {chunk}",
                                                          temperature=0,
                                                          top_p=None,
                                                          max_tokens=model_name_to_max_tokens("gpt-4.1-nano"),
                                                          default_model="gpt-4.1-nano",
                                                          default_provider=extract_model_provider("gpt-4.1-nano"),
                                                          structed_class=GradeDocuments,
                                                          configurable=True)

        self.answer_llm = make_llm_chain(system_prompt_or_path="{llm_sys_prompt}",
                                         human_message="user's question:\n{question}\n\nreference documents:\n{documents}",
                                         temperature=0,
                                         top_p=None,
                                         max_tokens=model_name_to_max_tokens("gpt-4.1-mini"),
                                         default_model="gpt-4.1-mini",
                                         default_provider=extract_model_provider("gpt-4.1-mini"),
                                         configurable=True)

        self.groundness_check_llm = make_structed_llm_chain(system_prompt_or_path=load_string_from_txt("./sys_prompt_hub/04_self_rag/03_groundness_evalute.txt"),
                                                            human_message="retrieved documents : {chunk}\n\nLLM-generated answer : {answer}",
                                                            temperature=0,
                                                            top_p=None,
                                                            max_tokens=model_name_to_max_tokens("gpt-4.1-mini"),
                                                            default_model="gpt-4.1-mini",
                                                            default_provider=extract_model_provider("gpt-4.1-mini"),
                                                            structed_class=Groundednesss,
                                                            configurable=True)

        self.relevance_check_llm = make_structed_llm_chain(system_prompt_or_path=load_string_from_txt("./sys_prompt_hub/04_self_rag/04_relevance_evaluate.txt"),
                                                           human_message="user's question : {question}\n\nLLM-generated answer : {answer}",
                                                           temperature=0,
                                                           top_p=None,
                                                           max_tokens=model_name_to_max_tokens("gpt-4.1-mini"),
                                                           default_model="gpt-4.1-mini",
                                                           default_provider=extract_model_provider("gpt-4.1-mini"),
                                                           structed_class=GradeAnswer,
                                                           configurable=True)

        self.llm_query_rewrite = make_structed_llm_chain(system_prompt_or_path=load_string_from_txt("./sys_prompt_hub/04_self_rag/05_query_rewrite.txt"),
                                                         human_message="user's question : {question}",
                                                         temperature=0,
                                                         top_p=None,
                                                         max_tokens=model_name_to_max_tokens("gpt-4.1-nano"),
                                                         default_model="gpt-4.1-nano",
                                                         default_provider=extract_model_provider("gpt-4.1-nano"),
                                                         structed_class=QueryRewrite,
                                                         configurable=True)
        

        self.memory = MemorySaver()

    ### 노드 정의 ###
    def retrieve_documents(self, state : GraphState, config : RunnableConfig):
        question = state["question"]
        result_chunks = self.retriever.invoke(question)
        chat_history = state["chat_history"][-6:]

        # chat history
        if len(chat_history) == 0:
            return {"result_chunks" : result_chunks}
        else:
            for i in range(len(chat_history)):
                if i & 1:
                    result_chunks.append(Document(page_content=chat_history[i].content, metadata={"source" : "AI Generated Conversation History", "page" : "None"}))
                else:
                    continue
            return {"result_chunks" : result_chunks}
            
    def grade_documents(self, state : GraphState, config : RunnableConfig):
        question = state["question"]
        result_chunks = state["result_chunks"]

        grade_inputs = [{"question" : question, "chunk" : chunk.page_content} for chunk in result_chunks]
        grade_results = self.grade_document_llm.batch(grade_inputs, config={"max_concurrency": 4})

        for relevance, chunk in zip(grade_results, result_chunks):
            chunk.metadata["relevance"] = relevance.binary_score
        
        result_chunks = [chunk for chunk in result_chunks if chunk.metadata["relevance"] == "yes"]

        return {"result_chunks" : result_chunks}

    def llm_answer(self, state : GraphState, config : RunnableConfig):
        question = state["question"]
        result_chunks = state["result_chunks"]
        llm_sys_prompt = state["llm_sys_prompt"]

        documents = ""
        for chunk in result_chunks:
            documents += content_template.format(source=chunk.metadata["source"], page=str(chunk.metadata["page"]), content=chunk.page_content)

        answer = self.answer_llm.invoke({"llm_sys_prompt" : llm_sys_prompt, "question" : question, "documents" : documents})

        return {"answer" : answer}

    def transform_query(self, state : GraphState, config : RunnableConfig):
        question = state["question"]

        query_rewrite = self.llm_query_rewrite.invoke({"question" : question}).rewritten_query

        return {"question" : query_rewrite}

    def make_chat_history(self, state : GraphState, config : RunnableConfig):
        question = state["question"]
        answer = state["answer"]

        return {"chat_history" : [("user", question), ("assistant", answer)]}

    ### 조건부 엣지 정의 ###
    def decide_to_generate(self, state : GraphState, config : RunnableConfig):
        filtered_documents = state["result_chunks"]

        if not filtered_documents:
            return "transform_query"
        else:
            return "llm_answer"

    def grade_generation_v_documents_and_question(self, state : GraphState, config : RunnableConfig):
        question = state["question"]
        result_chunks = state["result_chunks"]
        documents = ""
        for chunk in result_chunks:
            documents += content_template.format(source=chunk.metadata["source"], page=str(chunk.metadata["page"]), content=chunk.page_content)

        answer = state["answer"]

        grade = self.groundness_check_llm.invoke({"chunk" : documents, "answer" : answer}).binary_score

        if grade == "yes":
            grade = self.relevance_check_llm.invoke({"question" : question, "answer" : answer}).binary_score

            if grade == "yes":
                return "relevant"
            else:
                return "not relevant"
        else:
            return "hallucination"

    def make_app_graph(self):
        builder = StateGraph(GraphState, config_schema=GraphConfig)

        # 노드 정의
        builder.add_node("retrieve_documents", self.retrieve_documents)  # retrieve
        builder.add_node("grade_documents", self.grade_documents)  # grade documents
        builder.add_node("llm_answer", self.llm_answer)  # generatae
        builder.add_node("transform_query", self.transform_query)  # transform_query
        builder.add_node("make_chat_history", self.make_chat_history)

        # 엣지 정의
        builder.add_edge(START, "retrieve_documents")
        builder.add_edge("retrieve_documents", "grade_documents")

        builder.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "transform_query": "transform_query",
                "llm_answer": "llm_answer",
            },
        )

        builder.add_edge("transform_query", "retrieve_documents")

        # 답변 생성 노드에서 조건부 엣지 추가
        builder.add_conditional_edges(
            "llm_answer",
            self.grade_generation_v_documents_and_question,
            {
                "hallucination": "llm_answer",
                "relevant": "make_chat_history",
                "not relevant": "transform_query",
            },
        )

        builder.add_edge("make_chat_history", END)

        graph = builder.compile(checkpointer=self.memory)

        return graph