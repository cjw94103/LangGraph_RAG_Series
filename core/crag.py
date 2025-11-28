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

from langchain_teddynote.tools.tavily import TavilySearch
from utils.tavily_websearch_document_loader import postprocess_tavily

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

    chat_history : Annotated[list, add_messages]

    llm_sys_prompt : str

class GraphConfig(TypedDict, total=False):
    answer_llm_name : str
    answer_llm_temperature : float

class RetrieveEvaluate(BaseModel):
    """A system that evaluates the relevance score of retrieved chunks to the user's question."""
    
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="A relevance score between 0.0 and 1.0"
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

class CRAG:
    def __init__(self, retriever, confidence_threshold=0.7, top_k=5):
        self.retriever = retriever
        self.retriever.search_kwargs["k"] = top_k
        
        self.confidence_threshold = confidence_threshold

        self.llm_retrieve_eval = make_structed_llm_chain(system_prompt_or_path=load_string_from_txt("./sys_prompt_hub/03_crag/01_retrieval_evaluator.txt"),
                                                         human_message="user's question : {question}\n\nretrieved document : {chunk}",
                                                         temperature=0,
                                                         top_p=None,
                                                         max_tokens=model_name_to_max_tokens("gpt-4.1-nano"),
                                                         default_model="gpt-4.1-nano",
                                                         default_provider=extract_model_provider("gpt-4.1-nano"),
                                                         structed_class=RetrieveEvaluate,
                                                         configurable=True)

        self.llm_query_rewrite = make_structed_llm_chain(system_prompt_or_path=load_string_from_txt("./sys_prompt_hub/03_crag/02_query_rewrite.txt"),
                                                         human_message="user's question : {question}",
                                                         temperature=0,
                                                         top_p=None,
                                                         max_tokens=model_name_to_max_tokens("gpt-4.1-nano"),
                                                         default_model="gpt-4.1-nano",
                                                         default_provider=extract_model_provider("gpt-4.1-nano"),
                                                         structed_class=QueryRewrite,
                                                         configurable=True)

        self.answer_llm = make_llm_chain(system_prompt_or_path="{llm_sys_prompt}",
                                        human_message="user's question:\n{question}\n\nreference documents:\n{documents}\n\nchat_history:\n{chat_history}",
                                        temperature=0,
                                        top_p=None,
                                        max_tokens=model_name_to_max_tokens("gpt-4.1-mini"),
                                        default_model="gpt-4.1-mini",
                                        default_provider=extract_model_provider("gpt-4.1-mini"),
                                        configurable=True)

        self.tavily_tool = TavilySearch(
            topic="general",
            max_results=top_k,
            search_depth="basic", 
            include_answer=False,
            include_raw_content="text", 
            include_images=False,
            exclude_domains=["pinterest.com", "reddit.com"], 
            include_domains=[])

        
        self.memory = MemorySaver()

    def retrieve_documents(self, state : GraphState, config : RunnableConfig):
        question = state["question"]
        result_chunks = self.retriever.invoke(question)

        return {"result_chunks" : result_chunks}

    def retrieve_evaluate(self, state : GraphState, config : RunnableConfig):
        question = state["question"]
        result_chunks = state["result_chunks"]

        eval_inputs = [{"question" : question, "chunk" : chunk.page_content} for chunk in result_chunks]
        eval_results = self.llm_retrieve_eval.batch(eval_inputs, config={"max_concurrency": 4})


        for relevance_score, chunk in zip(eval_results, result_chunks):
            chunk.metadata["relevance_score"] = relevance_score.score

        result_chunks = [chunk for chunk in result_chunks if chunk.metadata["relevance_score"] > self.confidence_threshold]

        return {"result_chunks" : result_chunks}

    def web_search(self, state : GraphState, config : RunnableConfig):
        question = state["question"]
        search_query = self.llm_query_rewrite.invoke({"question" : question}).rewritten_query
        
        res = self.tavily_tool.search(search_query)

        return {"result_chunks" : postprocess_tavily(res)}

    def route_branch(self, state : GraphState, config : RunnableConfig):
        result_chunks = state["result_chunks"]

        if len(result_chunks) > 0 :
            return "llm_answer"
        elif len(result_chunks) == 0:
            return "web_search"

    def llm_answer(self, state : GraphState, config : RunnableConfig):
        question = state["question"]
        result_chunks = state["result_chunks"]
        chat_history = messages_to_history(state["chat_history"])

        documents = ""
        for chunk in result_chunks:
            documents += content_template.format(source=chunk.metadata["source"], page=str(chunk.metadata["page"]), content=chunk.page_content)

        llm_sys_prompt = state["llm_sys_prompt"]
        answer_llm_name = config.get("configurable", {}).get("answer_llm_name")
        answer_llm_temperature = config.get("configurable", {}).get("answer_llm_temperature")

        # streaming 대상만 이 코드를 사용
        chain = self.answer_llm.with_config(configurable={"model": answer_llm_name, 
                                                         "model_provider": extract_model_provider(answer_llm_name),
                                                         "max_tokens" : model_name_to_max_tokens(answer_llm_name),
                                                         "temperature" : answer_llm_temperature})

        answer = chain.invoke({"llm_sys_prompt": llm_sys_prompt, "question" : question, "documents" : documents, "chat_history" : chat_history})

        return {"answer" : answer, "chat_history" : [("user", question), ("assistant", answer)]}

    def make_app_graph(self):
        builder = StateGraph(GraphState, config_schema=GraphConfig)

        # 노드 추가
        builder.add_node("retrieve_documents", self.retrieve_documents)
        builder.add_node("retrieve_evaluate", self.retrieve_evaluate)
        builder.add_node("web_search", self.web_search)
        builder.add_node("llm_answer", self.llm_answer)

        # 엣지 연결
        builder.add_edge(START, "retrieve_documents")
        builder.add_edge("retrieve_documents", "retrieve_evaluate")

        builder.add_conditional_edges(
            "retrieve_evaluate",
            self.route_branch,
            {
                "llm_answer" : "llm_answer",
                "web_search" : "web_search"
            }
        )
        builder.add_edge("web_search", "llm_answer")
        builder.add_edge("llm_answer", END)

        graph = builder.compile(checkpointer=self.memory)

        return graph