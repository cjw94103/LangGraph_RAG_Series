from utils.llm_chain_factory import make_llm_chain, make_structed_llm_chain

from typing import Annotated, TypedDict, List, Literal, Any
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_teddynote.messages import messages_to_history
from utils.helper import extract_model_provider, model_name_to_max_tokens, load_string_from_txt
from langchain_core.runnables import RunnableConfig
from langchain.schema import Document
from pydantic import BaseModel, Field

from core.long_rag import map_chunks_to_original_pages

class GradeDocuments(BaseModel):
    score: str = Field(
        description="You must evaluate the relevance and completeness of the search results and output ONLY ONE of these three strings: 'correct' or 'ambiguous' or 'incorrect'"
    ) 


class GraphState(TypedDict):
    question : str
    answer : str
    answer_contexts : str
    
    result_chunks : List[Document]
    processed_chunks : List[Document]

    chat_history : Annotated[list, add_messages]

    llm_sys_prompt : str

class GraphConfig(TypedDict, total=False):
    llm_name : str
    temperature : float

class SelfRoute:
    def __init__(self, retriever, top_k=5):
        self.retriever = retriever
        self.retriever.search_kwargs["k"] = top_k
        self.vectorstore = retriever.vectorstore

        self.route_llm = make_structed_llm_chain(system_prompt_or_path=load_string_from_txt("./sys_prompt_hub/09_self_route/01_route_prompt.txt"),
                                                 human_message="User's question : {question}\n\nRetrieved document : {document}",
                                                 temperature=0,
                                                 top_p=None,
                                                 max_tokens=model_name_to_max_tokens("gpt-4.1-nano"),
                                                 default_model="gpt-4.1-nano",
                                                 default_provider=extract_model_provider("gpt-4.1-nano"),
                                                 structed_class=GradeDocuments,
                                                 configurable=True)
        
        self.llm_chain = make_llm_chain(system_prompt_or_path=load_string_from_txt("./sys_prompt_hub/09_self_route/02_answer_llm_prompt.txt"),
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
        retrieve_results = self.retriever.invoke(question)

        return {"result_chunks" : retrieve_results}

    def preprocess_documents(self, state : GraphState, config : RunnableConfig):
        question = state["question"]
        result_chunks = state["result_chunks"]

        inputs = [{"question" : question, "document" : chunk.page_content} for chunk in result_chunks]
        route_results = self.route_llm.batch(inputs, config={"max_concurrency" : 4})

        score_list = route_results  
        process_chunks = []
        
        for chunk, score in zip(result_chunks, score_list):
            score = score.score
        
            if score == "correct":
                process_chunks.append(chunk)
            elif score == "ambiguous":
                _ambi_result = map_chunks_to_original_pages([chunk], self.vectorstore)[0]
                ambi_result = Document(page_content=_ambi_result["content"], metadata=_ambi_result["metadata"])
                process_chunks.append(ambi_result)
            else:
                continue

        answer_contexts = ""
        for i, context in enumerate(process_chunks):
            context_str = "metadata:\n\n{metadata}\n\ncontent:\n\n{content}".format(metadata=context.metadata, content=context.page_content)
            context_str += "\n\n------------------------------ document delimeter ------------------------------\n\n"
            answer_contexts += context_str

        return {"processed_chunks" : process_chunks, "answer_contexts" : answer_contexts}

    def llm_answer(self, state : GraphState, config : RunnableConfig):
        question = state["question"]
        answer_contexts = state["answer_contexts"]
        chat_history = messages_to_history(state["chat_history"])
        
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
        builder.add_node("preprocess_documents", self.preprocess_documents)
        builder.add_node("llm_answer", self.llm_answer)

        # 엣지 연결
        builder.add_edge(START, "retrieve_documents")
        builder.add_edge("retrieve_documents", "preprocess_documents")
        builder.add_edge("preprocess_documents", "llm_answer")
        builder.add_edge("llm_answer", END)

        graph = builder.compile(checkpointer=self.memory)

        return graph