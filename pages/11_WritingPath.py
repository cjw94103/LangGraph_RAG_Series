import streamlit as st
import time

from utils.helper import openai_api_setting, tavily_api_setting, load_string_from_txt

from langchain_core.runnables import RunnableConfig
from core.writing_path import WritingPath

from utils.web_design import set_web_design

## Web Design
set_web_design(page_title="LangGraph RAG Algorithms", 
               page_icon="./logo_imgs/logo.png", 
               title="WritingPath", 
               caption="",
               logo_path = './logo_imgs/logo_page.png')

with st.sidebar:
    "이 기법은 Report 또는 Article 형식의 문서를 작성할 때 사용할 수 있습니다."
    "RAG 기법의 Extension이며, 웹 검색 기반의 외부 자료를 기반으로 동작합니다."
    "주요 파라미터는 API 비용을 감안하여, 고정된 값을 사용하며, 상세한 파라미터는 core/writing_path.py에서 확인해보실 수 있습니다."
    "현재는 문서의 일부만 작성하도록 세팅되어 있습니다. 문서의 전부 생성을 원하실 경우 core/writing_path.py에서 make_report 함수의 break를 해제하세요."

    "😀 메타 데이터 입력 😀"

    st.divider()
    purpose = st.text_input("문서 작성의 목적을 입력해주세요. (예시 : 삼성전자가 만든 인공지능 가우스에 대한 설명)")
    st.divider()
    writing_type = st.text_input("문서의 특징을 입력해주세요. (예시 : 전자 제품 전문 신문사에서 작성한 기사)")
    st.divider()
    style = st.text_input("문서의 톤앤매너를 설정합니다. (예시 : 알아듣기 쉽고 신문 기사 처럼)")
    st.divider()
    openai_api_key = st.text_input("OpenAI API Key를 입력하세요", type="password")
    tavily_api_key = st.text_input("Tavily API Key를 입력하세요", type="password")
    st.divider()

    process = st.button("Process")

    ## save global parameter
    st.session_state.purpose = purpose
    st.session_state.style = style
    st.session_state.writing_type = writing_type

if process:
    ## OpenAI API 설정
    openai_api_setting(openai_api_key)
    tavily_api_setting(tavily_api_key)

    ## graph 생성
    graph_ins = WritingPath(top_k=1)
    app = graph_ins.make_app_graph()

    st.session_state.app = app

    input_template = """
    **User Input:**
    - Purpose of the Writing: {purpose}
    - Type of Writing: {writing_type}
    - Style of Writing: {style}
        """

    input_metadata = input_template.format(purpose=st.session_state.purpose, 
                                           writing_type=st.session_state.writing_type, 
                                           style=st.session_state.style)
    config = RunnableConfig(recursion_limit=20, configurable={"llm_name" : "gpt-4.1",
                                                          "temperature" : 0.2})
    inputs = {"input_metadata" : input_metadata,
              "answer" : "",
              "title" : "",
              "header" : [],
              "keywords" : [],
              "keywords_concepts" : [],
              "keywords_add_insights" : [],
              "subheader" : [],
              "result_temp" : []}

    with st.spinner("문서를 작성중입니다............"):
        result = app.invoke(inputs, config=config)

    st.write(result["answer"])