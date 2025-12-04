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

class AgentAnswerFormat(BaseModel):
    answer: str = Field(description="Answer the question based on the document and other agents’ responses. Provide your answer")
    explanation: str = Field(description="step-by-step reasoning explanation")

class AggregatorAnswerFormat(BaseModel):
    correct_answer: str = Field(description="please provide all possible correct answers")
    explanation: str = Field(description="also provide a step-by-step reasoning explanation")

class GraphConfig(TypedDict, total=False):
    answer_llm_name : str
    answer_llm_temperature : float
    
class GraphState(TypedDict):
    question : str
    answer : str

    agents_res_list : List[str]
    aggre_history : str

    result_chunks : List[Document]
    chat_history : Annotated[list, add_messages]

    # number of loops
    loop_count : int

class MADAM_RAG:
    def __init__(self, retriever, top_k=5, round_T=3):
        self.retriever = retriever
        self.retriever.search_kwargs["k"] = top_k
        self.round_T = round_T

        self.res_agent_llm = make_structed_llm_chain(system_prompt_or_path="",
                                                      human_message=load_string_from_txt("./sys_prompt_hub/07_madam_rag/01_res_agent_prompt.txt"),
                                                      temperature=0,
                                                      top_p=None,
                                                      max_tokens=model_name_to_max_tokens("gpt-4.1-nano"),
                                                      default_model="gpt-4.1-nano",
                                                      default_provider=extract_model_provider("gpt-4.1-nano"),
                                                      structed_class=AgentAnswerFormat,
                                                      configurable=True)
        
        self.aggregator_llm = make_structed_llm_chain(system_prompt_or_path=load_string_from_txt("./sys_prompt_hub/07_madam_rag/02_aggregator_prompt.txt"),
                                                      human_message="Question: {question}\nAgent responses: {agent_reponse_list}",
                                                      temperature=0,
                                                      top_p=None,
                                                      max_tokens=model_name_to_max_tokens("gpt-4.1-nano"),
                                                      default_model="gpt-4.1-nano",
                                                      default_provider=extract_model_provider("gpt-4.1-nano"),
                                                      structed_class=AggregatorAnswerFormat,
                                                      configurable=True)

        self.final_answer_llm = make_llm_chain(system_prompt_or_path=load_string_from_txt("./sys_prompt_hub/07_madam_rag/03_final_answer_llm_prompt.txt"),
                                              human_message="User's Question : {question}\n\nCorrect answer and Reasoning explanation : {aggre_history}",
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

    def intermediate_response(self, state : GraphState, config : RunnableConfig):
        question = state["question"]
        result_chunks = state["result_chunks"]
        aggre_history = state["aggre_history"]

        if len(aggre_history) == 0:
            aggre_history = ""

        inter_res_inputs = [{"question" : question, 
                             "document" : content_template.format(source=chunk.metadata["source"],
                                                                 page=str(chunk.metadata["page"]),
                                                                 content=chunk.page_content),
                            "history" : aggre_history} for chunk in result_chunks]
        inter_res_output = self.res_agent_llm.batch(inter_res_inputs, config={"max_concurrency": 4})
        inter_res_output = ["Answer: {answer}\n\nExplanation : {explanation}".format(answer=output.answer, 
                                                                                     explanation=output.explanation) for output in inter_res_output]

        return {"agents_res_list" : inter_res_output}

    def aggregate_response(self, state : GraphState, config : RunnableConfig):
        question = state["question"]
        agents_res_list = state["agents_res_list"]

        res_str = ""
        for i, res in enumerate(agents_res_list):
            res_str += "Agent {idx}\n\n{answer}\n\n".format(idx = str(i+1), answer=res)

        aggre_result = self.aggregator_llm.invoke({"question" : question, "agent_reponse_list" : res_str})
        aggre_result = "Correct Answer: {answer}\n\nExplanation : {explanation}".format(answer=aggre_result.correct_answer, 
                                                                                        explanation=aggre_result.explanation)

        # loop 증가
        loop_count = state["loop_count"]
        loop_count += 1

        return {"aggre_history" : aggre_result, "loop_count" : loop_count}

    def llm_answer(self, state : GraphState, config : RunnableConfig):
        question = state["question"]
        aggre_history = state["aggre_history"]

        answer_llm_name = config.get("configurable", {}).get("answer_llm_name")
        answer_llm_temperature = config.get("configurable", {}).get("answer_llm_temperature")
        
        # streaming 대상만 이 코드를 사용
        chain = self.final_answer_llm.with_config(configurable={"model": answer_llm_name, 
                                                                 "model_provider": extract_model_provider(answer_llm_name),
                                                                 "max_tokens" : model_name_to_max_tokens(answer_llm_name),
                                                                 "temperature" : answer_llm_temperature})
        
        final_answer = chain.invoke({"question" : question, "aggre_history" : aggre_history})

        return {"answer" : final_answer, "chat_history" : [("user", question), ("assistant", final_answer)]}
        
    # 조건부 엣지
    def decide_to_loop_end(self, state : GraphState, config : RunnableConfig):
        loop_count = state["loop_count"]

        if loop_count == self.round_T:
            return "llm_answer"
        else:
            return "intermediate_response"

    def make_app_graph(self):
        builder = StateGraph(GraphState, config_schema=GraphConfig)

        # 노드 추가
        builder.add_node("retrieve_documents", self.retrieve_documents)
        builder.add_node("intermediate_response", self.intermediate_response)
        builder.add_node("aggregate_response", self.aggregate_response)
        builder.add_node("llm_answer", self.llm_answer)
        
        # 엣지 연결
        builder.add_edge(START, "retrieve_documents")
        builder.add_edge("retrieve_documents", "intermediate_response")
        builder.add_edge("intermediate_response", "aggregate_response")

        builder.add_conditional_edges(
            "aggregate_response",
            self.decide_to_loop_end,
            {
                "llm_answer": "llm_answer",
                "intermediate_response": "intermediate_response",
            },
        )
        
        builder.add_edge("llm_answer", END)

        graph = builder.compile(checkpointer=self.memory)

        return graph