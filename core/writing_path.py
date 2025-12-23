from utils.llm_chain_factory import make_llm_chain, make_structed_llm_chain

from langchain_teddynote.tools.tavily import TavilySearch
from typing import Annotated, TypedDict, List, Literal, Any
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_teddynote.messages import messages_to_history
from utils.helper import extract_model_provider, model_name_to_max_tokens, load_string_from_txt
from langchain_core.runnables import RunnableConfig
from langchain.schema import Document
from pydantic import BaseModel, Field
from utils.tavily_websearch_document_loader import postprocess_tavily

## content template
content_template = """source : {source}
page : {page}
content " {content}
------------------------------------delimeter------------------------------------"""

class TitleOutlineOutput(BaseModel):
    """Title and Outline structure for document generation"""
    title: str = Field(description="The main title of the document")
    outline: List[str] = Field(description="List of section headers/outlines, formatted as numbered items (e.g., '1. Introduction', '2. Main Topic')")

class KeywordExtractionOutput(BaseModel):
    """Extracted keywords from external documents"""
    keywords: List[str] = Field(
        description="List of core keywords and key phrases extracted from the document"
    )
    key_concepts: List[str] = Field(
        description="List of important concepts and topics identified in the document"
    )
    additional_insights: List[str] = Field(
        description="List of supplementary information or insights that can enrich the outline"
    )

class SubHeaderOutput(BaseModel):
    """Sub-headers generated for a specific header section"""
    header: str = Field(description="The original header section")
    subheaders: List[str] = Field(
        description="List of sub-headers that provide detailed breakdown of the main header section"
    )

class QueryRewriteOutput(BaseModel):
    """Rewritten query optimized for web search"""
    rewritten_query: str = Field(
        description="Search-optimized query rewritten from the title and subheader"
    )

class ContentOutput(BaseModel):
    """Generated content for a subheader section with markdown formatting"""
    subheader: str = Field(description="The subheader with appropriate markdown heading level (### or ####)")
    content: str = Field(description="Generated content with markdown formatting including bullet points, bold, italic, etc.")
    sources_used: bool = Field(description="Whether external search results were used")
    
class GraphState(TypedDict):
    input_metadata : str

    title : str
    header : List[str]
    keywords : List[str]
    keywords_concepts : List[str]
    keywords_add_insights : List[str]
    subheader : List[dict]

    answer : str
    # answer_contexts : str
    # summary_contexts : str

    # chat_history : Annotated[list, add_messages]

    result_temp : list

class GraphConfig(TypedDict, total=False):
    llm_name : str
    temperature : float

class WritingPath:
    def __init__(self, top_k=3):
        self.tavily_tool = TavilySearch(
            topic="general",
            max_results=top_k,
            search_depth="basic", 
            include_answer=False,
            include_raw_content="text", 
            include_images=False,
            exclude_domains=["pinterest.com", "reddit.com"], 
            include_domains=[])

        self.init_title_outline_llm = make_structed_llm_chain(system_prompt_or_path=load_string_from_txt("./sys_prompt_hub/11_writing_path/01_init_title_outline_sys_prompt.txt"),
                                                             human_message="{input_metadata}",
                                                             temperature=0,
                                                             top_p=None,
                                                             max_tokens=model_name_to_max_tokens("gpt-4.1-mini"),
                                                             default_model="gpt-4.1-mini",
                                                             default_provider=extract_model_provider("gpt-4.1-mini"),
                                                             structed_class=TitleOutlineOutput,
                                                             configurable=True)
        
        self.keyword_extract_llm = make_structed_llm_chain(system_prompt_or_path=load_string_from_txt("./sys_prompt_hub/11_writing_path/02_keyword_extract_sys_prompt.txt"),
                                                             human_message="{documents}",
                                                             temperature=0,
                                                             top_p=None,
                                                             max_tokens=model_name_to_max_tokens("gpt-4.1-mini"),
                                                             default_model="gpt-4.1-mini",
                                                             default_provider=extract_model_provider("gpt-4.1-mini"),
                                                             structed_class=KeywordExtractionOutput,
                                                             configurable=True)
        
        self.subheader_llm = make_structed_llm_chain(system_prompt_or_path=load_string_from_txt("./sys_prompt_hub/11_writing_path/03_01_subheader_hierarchy_second_depth_sys_prompt.txt"),
                                                     human_message="Title: {title}\nHeader: {header}\nKeywords: {keywords}",
                                                     temperature=0,
                                                     top_p=None,
                                                     max_tokens=model_name_to_max_tokens("gpt-4.1"),
                                                     default_model="gpt-4.1",
                                                     default_provider=extract_model_provider("gpt-4.1"),
                                                     structed_class=SubHeaderOutput,
                                                     configurable=True)
        
        self.query_rewrite_llm = make_structed_llm_chain(system_prompt_or_path=load_string_from_txt("./sys_prompt_hub/11_writing_path/04_01_query_rewrite_sys_prompt.txt"),
                                                         human_message="Title: {title}\nSubHeader: {subheader}",
                                                         temperature=0,
                                                         top_p=None,
                                                         max_tokens=model_name_to_max_tokens("gpt-4.1-nano"),
                                                         default_model="gpt-4.1-mini",
                                                         default_provider=extract_model_provider("gpt-4.1-nano"),
                                                         structed_class=QueryRewriteOutput,
                                                         configurable=True)

        self.content_generation_llm = make_structed_llm_chain(system_prompt_or_path=load_string_from_txt("./sys_prompt_hub/11_writing_path/04_02_content_generation_sys_prompt.txt"),
                                                             human_message="Retrieved Documents:\n{documents}\nSubHeader: {subheader}",
                                                             temperature=0,
                                                             top_p=None,
                                                             max_tokens=model_name_to_max_tokens("gpt-4.1"),
                                                             default_model="gpt-4.1",
                                                             default_provider=extract_model_provider("gpt-4.1"),
                                                             structed_class=ContentOutput,
                                                             configurable=True)

    def init_title_outline_generate(self, state : GraphState, config : RunnableConfig):
        input_metadata = state["input_metadata"]
        title_outline_result = self.init_title_outline_llm.invoke({"input_metadata" : input_metadata})

        title = title_outline_result.title
        header = title_outline_result.outline

        return {"title" : title, "header" : header}

    def keyword_extract(self, state : GraphState, config : RunnableConfig):
        title = state["title"]

        search_results = self.tavily_tool.search(title)
        search_results = postprocess_tavily(search_results)

        documents = ""
        for chunk in search_results:
            documents += content_template.format(source=chunk.metadata["source"], page=str(chunk.metadata["page"]), content=chunk.page_content)

        keyword_extract_result = self.keyword_extract_llm.invoke({"documents" : documents})

        return {"keywords" : keyword_extract_result.keywords, 
                "keywords_concepts" : keyword_extract_result.key_concepts,
               "keywords_add_insights" : keyword_extract_result.additional_insights}
        
    def make_sub_header(self, state : GraphState, config : RunnableConfig):
        title = state["title"]
        header = state["header"]
        keywords = ""
        for keyword in state["keywords"]:
            keywords += keyword + ", "

        inputs = [{"title" : title, "header" : header_text, "keywords" : keywords} for header_text in header]
        subheader_results = self.subheader_llm.batch(inputs, config={"max_concurrency" : 8})

        
        subheader_list = []
        for result in subheader_results:
            subheader_dict = {}
            subheader_dict[result.header] = result.subheaders
            subheader_list.append(subheader_dict)
        
        return {"subheader" : subheader_list}
        
    def make_report(self, state : GraphState, config : RunnableConfig):
        title = state["title"]
        subheader_list = state["subheader"]

        final_report_text = "# " + title + "\n\n" ## title 맨 처음에 추가
        for sub in subheader_list:
            final_report_text += "\n## " + list(sub.keys())[0] + "\n\n" # header 추가

            # 웹 검색을 위한 query rewrite
            target_subheader_list = [subheader for subheader in sub[list(sub.keys())[0]]]
            query_rewrite_inputs = [{"title" : title, "subheader" : subheader} for subheader in target_subheader_list]
            query_rewrite_batch_results = self.query_rewrite_llm.batch(query_rewrite_inputs, config={"max_concurrency" : 8})

            # 재작성된 쿼리 웹 검색 batch 준비
            queries = [query_rewrite_batch_result.rewritten_query for query_rewrite_batch_result in query_rewrite_batch_results]

            # batch 웹 검색
            web_results = self.tavily_tool.batch(queries)
            web_results_text_list = []

            for web_res in web_results:
                web_res = postprocess_tavily(web_res)
                documents = ""
                for chunk in web_res:
                    documents += content_template.format(source=chunk.metadata["source"], page=str(chunk.metadata["page"]), content=chunk.page_content)
                web_results_text_list.append(documents)
            
            # 웹 검색 기록 기반 답변 생성
            content_inputs = [{"documents" : documents, "subheader" : sub_header} for sub_header, web_result in zip(target_subheader_list, web_results_text_list)]
            content_batch_results = self.content_generation_llm.batch(content_inputs, config={"max_concurrency" : 8})

            for content in content_batch_results:
                final_report_text += content.subheader + "\n\n"
                final_report_text += content.content + "\n\n"
        
            break # 모든 문서 작성을 원할때 break 해제

        return {"answer" : final_report_text}


    def make_app_graph(self):
        builder = StateGraph(GraphState, config_schema=GraphConfig)

        # 노드 추가
        builder.add_node("init_title_outline_generate", self.init_title_outline_generate)
        builder.add_node("keyword_extract", self.keyword_extract)
        builder.add_node("make_sub_header", self.make_sub_header)
        builder.add_node("make_report", self.make_report)

        # 엣지 연결
        builder.add_edge(START, "init_title_outline_generate")
        builder.add_edge("init_title_outline_generate", "keyword_extract")
        builder.add_edge("keyword_extract", "make_sub_header")
        builder.add_edge("make_sub_header", "make_report")
        builder.add_edge("make_report", END)

        graph = builder.compile()

        return graph