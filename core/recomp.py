from utils.llm_chain_factory import make_llm_chain, make_structed_llm_chain

from typing import Annotated, TypedDict, List, Literal, Any
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_teddynote.messages import messages_to_history
from utils.helper import extract_model_provider, model_name_to_max_tokens, load_string_from_txt
from langchain_core.runnables import RunnableConfig
from langchain.schema import Document

class GraphState(TypedDict):
    question : str
    answer : str
    answer_contexts : str
    summary_contexts : str

    chat_history : Annotated[list, add_messages]

class GraphConfig(TypedDict, total=False):
    llm_name : str
    temperature : float

class RECOMP:
    def __init__(self, retriever, top_k=5):
        self.retriever = retriever
        self.retriever.search_kwargs["k"] = top_k

        self.sum_llm = make_llm_chain(system_prompt_or_path=load_string_from_txt("./sys_prompt_hub/10_recomp/01_summary_llm_prompt.txt"),
                                      human_message="user's question:\n{question}\n\nreference documents:\n{documents}",
                                      temperature=0,
                                      top_p=None,
                                      max_tokens=model_name_to_max_tokens("gpt-4.1-mini"),
                                      default_model="gpt-4.1-mini",
                                      default_provider=extract_model_provider("gpt-4.1-mini"),
                                      configurable=True)
        
        self.llm_chain = make_llm_chain(system_prompt_or_path=load_string_from_txt("./sys_prompt_hub/10_recomp/02_answer_llm_prompt.txt"),
                                       human_message="user's question:\n{question}\n\nreference summary documents:\n{documents}",
                                       temperature=0,
                                       top_p=None,
                                       max_tokens=model_name_to_max_tokens("gpt-4.1-mini"),
                                       default_model="gpt-4.1-mini",
                                       default_provider=extract_model_provider("gpt-4.1-mini"),
                                       configurable=True)

        self.memory = MemorySaver()

    def retrieve_documents(self, state : GraphState, config : RunnableConfig):
        question = state["question"]
        retrieve_results = self.retriever.invoke(question)

        answer_contexts = ""
        for i, context in enumerate(retrieve_results):
            context_str = "metadata:\n\n{metadata}\n\ncontent:\n\n{content}".format(metadata=context.metadata, content=context.page_content)
            context_str += "\n\n------------------------------ document delimeter ------------------------------\n\n"
            answer_contexts += context_str

        return {"answer_contexts" : answer_contexts}

    def summarize_documents(self, state : GraphState, config : RunnableConfig):
        question = state["question"]
        contexts = state["answer_contexts"]

        summary_contexts = self.sum_llm.invoke({"question" : question, "documents" : contexts})

        return {"summary_contexts" : summary_contexts}

    def llm_answer(self, state : GraphState, config : RunnableConfig):
        question = state["question"]
        answer_contexts = state["summary_contexts"]

        llm_name = config.get("configurable", {}).get("llm_name")
        temperature = config.get("configurable", {}).get("temperature")

        # streaming 대상만 이 코드를 사용
        chain = self.llm_chain.with_config(configurable={"model": llm_name, 
                                                         "model_provider": extract_model_provider(llm_name),
                                                         "max_tokens" : model_name_to_max_tokens(llm_name),
                                                         "temperature" : temperature})

        answer = chain.invoke({"question" : question, "documents" : answer_contexts})

        return {"answer" : answer, "chat_history" : [("user", question), ("assistant", answer)]}

    def make_app_graph(self):
        builder = StateGraph(GraphState, config_schema=GraphConfig)

        # 노드 추가
        builder.add_node("retrieve_documents", self.retrieve_documents)
        builder.add_node("summarize_documents", self.summarize_documents)
        builder.add_node("llm_answer", self.llm_answer)

        # 엣지 연결
        builder.add_edge(START, "retrieve_documents")
        builder.add_edge("retrieve_documents", "summarize_documents")
        builder.add_edge("summarize_documents", "llm_answer")
        builder.add_edge("llm_answer", END)

        graph = builder.compile(checkpointer=self.memory)

        return graph