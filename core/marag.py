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

class PlanFormat(BaseModel):
    analysis: str = Field(description= "Your analysis. Think step-by-step")
    step: List[str] = Field(description= "different steps to follow, should be in sorted order")

class StepTaskFormat(BaseModel):
    type: str = Field(description="Type of task, one of [aggregate, question-answering]")
    task: str = Field(description="The detail task to do in this step")

class QAAnswerFormat(BaseModel):
    analysis: str = Field(description="Your thoughts, analysis about the question and the context. Think step-by-step")
    answer: str = Field(description="The answer for the question")
    success: str = Field(description="binary output (Yes or No), indicate you can answer or not")
    rating: int = Field( default=None, description="How confident, from 0 to 10. More evidence, more agreement, more confident")

class PlanSummaryFormat(BaseModel): 
    output: str = Field(description="your output, follow the format")
    answer: str = Field(description="Final answer for the question")
    score: int = Field(description="Confident score")

class GraphConfig(TypedDict, total=False):
    answer_llm_name : str
    answer_llm_temperature : float

class GraphState(TypedDict):
    question : str
    answer : str

    ## 누적 히스토리
    memory : str

    ## planner agent
    planning_step : List[str]
    planning_analysis : str

    ## step definer
    step_definer_type : str
    step_definer_task : str
    remain_planning_step : List[str]

    ## QA Agent
    qa_analysis : str
    qa_answer : str
    qa_success : str
    qa_rating : int

    ## extractor
    result_chunks : List[Document]
    filtered_passage : str
    
    chat_history : Annotated[list, add_messages]
    llm_sys_prompt : str 

class MARAG:
    def __init__(self, retriever, top_k=5):
        self.retriever = retriever
        self.retriever.search_kwargs["k"] = top_k

        self.planner_llm = make_structed_llm_chain(system_prompt_or_path=load_string_from_txt("./sys_prompt_hub/05_marag/01_planner_prompt.txt"),
                                                  human_message="user's question : {question}",
                                                  temperature=0,
                                                  top_p=None,
                                                  max_tokens=model_name_to_max_tokens("gpt-4.1-nano"),
                                                  default_model="gpt-4.1-nano",
                                                  default_provider=extract_model_provider("gpt-4.1-nano"),
                                                  structed_class=PlanFormat,
                                                  configurable=True)

        self.step_definer_llm = make_structed_llm_chain(system_prompt_or_path=load_string_from_txt("./sys_prompt_hub/05_marag/02_step_definer_prompt.txt"),
                                                        human_message="Plan : {plan}\nCurrent step : {cur_step}\nResults of finished steps : {memory}",
                                                        temperature=0,
                                                        top_p=None,
                                                        max_tokens=model_name_to_max_tokens("gpt-4.1-nano"),
                                                        default_model="gpt-4.1-nano",
                                                        default_provider=extract_model_provider("gpt-4.1-nano"),
                                                        structed_class=StepTaskFormat,
                                                        configurable=True)
        
        self.extractor_llm = make_llm_chain(system_prompt_or_path=load_string_from_txt("./sys_prompt_hub/05_marag/03_extractor_prompt.txt"),
                                            human_message="Passage:\n###\n{passage}\n###\n\nQuery: {question}?",
                                            temperature=0,
                                            top_p=None,
                                            max_tokens=model_name_to_max_tokens("gpt-4.1-nano"),
                                            default_model="gpt-4.1-nano",
                                            default_provider=extract_model_provider("gpt-4.1-nano"),
                                            configurable=True)

        self.qa_llm = make_structed_llm_chain(system_prompt_or_path=load_string_from_txt("./sys_prompt_hub/05_marag/04_qa_prompt.txt"),
                                              human_message="Retrieved documents:\n{context}\nQuestion: {question}",
                                              temperature=0,
                                              top_p=None,
                                              max_tokens=model_name_to_max_tokens("gpt-4.1-mini"),
                                              default_model="gpt-4.1-mini",
                                              default_provider=extract_model_provider("gpt-4.1-mini"),
                                              structed_class=QAAnswerFormat,
                                              configurable=True)

        self.final_answer_llm = make_structed_llm_chain(system_prompt_or_path=load_string_from_txt("./sys_prompt_hub/05_marag/05_final_answer_prompt.txt"),
                                                      human_message="Original Question : {question}\nPlan : {plan}\noutput of steps : \n{memory}",
                                                      temperature=0,
                                                      top_p=None,
                                                      max_tokens=model_name_to_max_tokens("gpt-4.1-mini"),
                                                      default_model="gpt-4.1-mini",
                                                      default_provider=extract_model_provider("gpt-4.1-mini"),
                                                      structed_class=PlanSummaryFormat,
                                                      configurable=True)

        self.memory = MemorySaver()

    ### 노드 정의 ###
    def planning(self, state : GraphState, config : RunnableConfig):
        question = state["question"]
        planning_result = self.planner_llm.invoke({"question" : question})

        return {"planning_step" : planning_result.step, 
                "planning_analysis" : planning_result.analysis, 
                "remain_planning_step" : planning_result.step}

    def step_define(self, state : GraphState, config : RunnableConfig):
        plan = state["planning_step"]
        remain_step = state["remain_planning_step"]
        cur_step = remain_step[0]
        memory = state["memory"]

        if len(memory) == 0:
            memory = "empty"

        step_definer_result = self.step_definer_llm.invoke({"plan" : plan, "cur_step" : cur_step, "memory" : "memory"})

        return {"step_definer_type" : step_definer_result.type, "step_definer_task" : step_definer_result.task}

    def retrieve_documents(self, state : GraphState, config : RunnableConfig):
        task = state["step_definer_task"]
        result_chunks = self.retriever.invoke(task)

        return {"result_chunks" : result_chunks}

    def document_extract(self, state : GraphState, config : RunnableConfig):
        question = state["step_definer_task"]
        result_chunks = state["result_chunks"]
        
        documents = ""
        for chunk in result_chunks:
            documents += content_template.format(source=chunk.metadata["source"], 
                                                 page=str(chunk.metadata["page"]), 
                                                 content=chunk.page_content)
        document_filter_result = self.extractor_llm.invoke({"passage" : documents, "question" : question})

        return {"filtered_passage" : document_filter_result}

    def step_answer(self, state : GraphState, config : RunnableConfig):
        question = state["step_definer_task"]
        context = state["filtered_passage"]
        remain_step = state["remain_planning_step"]

        # filling memory
        memory = state["memory"]
        plan = state["planning_step"]

        answer_step = self.qa_llm.invoke({"context" : context, "question" : question})

        memory += "Task : {task}\nQuestion : {question}\nAnswer : {answer}\nConfident Score : {rating}\n\n".format(task=remain_step[0],
                                                                                                               question=question,
                                                                                                               answer=answer_step.answer,
                                                                                                               rating=answer_step.rating)
    

        return {"qa_analysis" : answer_step.analysis, 
                "qa_answer" : answer_step.answer,
                "qa_success" : answer_step.success,
                "qa_rating" : answer_step.rating,
                "remain_planning_step" : remain_step[1:],
                "memory" : memory}

    def llm_answer(self, state : GraphState, config : RunnableConfig):
        question = state["question"]
        plan = str(state["planning_step"])
        memory = state["memory"]

        answer_llm_name = config.get("configurable", {}).get("answer_llm_name")
        answer_llm_temperature = config.get("configurable", {}).get("answer_llm_temperature")

        # streaming 대상만 이 코드를 사용
        chain = self.final_answer_llm.with_config(configurable={"model": answer_llm_name, 
                                                         "model_provider": extract_model_provider(answer_llm_name),
                                                         "max_tokens" : model_name_to_max_tokens(answer_llm_name),
                                                         "temperature" : answer_llm_temperature})

        final_answer = chain.invoke({"question" : question, "plan" : plan, "memory" : memory})
        
        
        return {"answer" : final_answer.answer, "chat_history" : [("user", question), ("assistant", final_answer.answer)]}

    ## 조건부 엣지
    def decide_to_final_answer(self, state : GraphState, config : RunnableConfig):
        step_definer_type = state["step_definer_type"]
        remain_planning_step = state["remain_planning_step"]

        if step_definer_type == "aggregate" or len(remain_planning_step) == 0:
            return "llm_answer"
        else:
            return "step_define"
        
    def make_app_graph(self):
        builder = StateGraph(GraphState, config_schema=GraphConfig)

        # 노드 정의
        builder.add_node("planning", self.planning)
        builder.add_node("step_define", self.step_define)
        builder.add_node("retrieve_documents", self.retrieve_documents)
        builder.add_node("document_extract", self.document_extract)
        builder.add_node("step_answer", self.step_answer)
        builder.add_node("llm_answer", self.llm_answer)
        

        # 엣지 정의
        builder.add_edge(START, "planning")
        builder.add_edge("planning", "step_define")
        builder.add_edge("step_define", "retrieve_documents")
        builder.add_edge("retrieve_documents", "document_extract")
        builder.add_edge("document_extract", "step_answer")

        builder.add_conditional_edges(
            "step_answer",
            self.decide_to_final_answer,
            {
                "llm_answer": "llm_answer",
                "step_define": "step_define",
            },
        )
        builder.add_edge("llm_answer", END)

        graph = builder.compile(checkpointer=self.memory)

        return graph