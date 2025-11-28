import streamlit as st
import time
import uuid

from utils.helper import openai_api_setting, make_uploadfile_to_retriever, load_string_from_txt, tavily_api_setting
from langchain_openai import OpenAIEmbeddings

from langchain_core.runnables import RunnableConfig
from core.crag import CRAG

from utils.web_design import set_web_design
from utils.upload_utils import save_cache_files

## Web Design
set_web_design(page_title="LangGraph RAG Algorithms", 
               page_icon="./logo_imgs/logo.png", 
               title="Corrective RAG", 
               caption="",
               logo_path = './logo_imgs/logo_page.png')

with st.sidebar:
    "😀 주요 파라미터 설정 😀"

    st.divider()
    llm_name = st.selectbox("사용 할 LLM 모델을 선택하세요.", ("gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"))
    st.divider()
    temperature = st.slider("Temperature 파라미터 설정", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    top_k = st.slider("Top-k 파라미터 (문서 검색 수) 설정", min_value=1, max_value=10, value=3, step=1)
    st.divider()
    uploaded_files = st.file_uploader("📄파일 업로드📄", type=['docx', 'pdf', 'hwpx', 'txt', 'md'], accept_multiple_files=True)
    st.divider()
    openai_api_key = st.text_input("OpenAI API Key를 입력하세요", type="password")
    tavily_api_key = st.text_input("Tavily Key를 입력하세요", type="password")
    st.divider()

    process = st.button("Process")
    st.divider()
    conver_init = st.button("대화 초기화", type="primary")

    ## save global parameter
    st.session_state.llm_name = llm_name
    st.session_state.temperature = temperature
    st.session_state.top_k = top_k

if process:
    ## OpenAI API 설정
    openai_api_setting(openai_api_key)
    tavily_api_setting(tavily_api_key)

    ## 시스템 프롬프트
    st.session_state.llm_sys_prompt = load_string_from_txt("./sys_prompt_hub/03_crag/03_llm_answer_system_message.txt")

    ## 업로드 파일 처리
    with st.spinner("업로드한 파일을 처리하고 있습니다...", show_time=True):
        filepaths = save_cache_files(directory="./user_upload_files", uploaded_files=uploaded_files)

    # dense retriever
    retriever = make_uploadfile_to_retriever(filepaths=filepaths, 
                                             embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
                                             embedding_dim=1536,
                                             top_k=st.session_state.top_k)

    st.session_state.retriever = retriever

    ## graph 생성
    thread_id = str(uuid.uuid4())
    graph_ins = CRAG(retriever=st.session_state.retriever, top_k=st.session_state.top_k)
    app = graph_ins.make_app_graph()

    st.session_state.thread_id = thread_id
    st.session_state.app = app

if conver_init:
    st.session_state.pop('app', None)
    st.session_state.pop('messages', None)

    thread_id = str(uuid.uuid4())
    graph_ins = CRAG(retriever=st.session_state.retriever, top_k=st.session_state.top_k)
    app = graph_ins.make_app_graph()

    st.session_state.thread_id = thread_id
    st.session_state.app = app

## 대화 시작
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "무엇을 도와드릴까요?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if question := st.chat_input():
    with st.chat_message("user"):
        st.session_state.messages.append({"role": "user", "content": question})
        st.write(question)

    with st.chat_message("assistant"):
        config = RunnableConfig(recursion_limit=20, configurable={"thread_id": st.session_state.thread_id, 
                                                                  "answer_llm_name" : st.session_state.llm_name,
                                                                  "answer_llm_temperature" : st.session_state.temperature})
        
        inputs = {"question" : question,
                 "answer" : "",
                 "result_chunks" : [],
                 "chat_history" : [],
                 "llm_sys_prompt" : st.session_state.llm_sys_prompt}

        output_str = ""
        def stream_data():
            global output_str
            for chunk_msg, metadata  in st.session_state.app.stream(inputs, config, stream_mode="messages"):
                if metadata["langgraph_node"] == "llm_answer":
                    if chunk_msg.content:
                        output_str += chunk_msg.content
                        yield chunk_msg.content
                        time.sleep(0.02)
                        
        st.write_stream(stream_data)
        st.session_state.messages.append({"role": "assistant", "content": output_str})