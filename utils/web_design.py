import streamlit as st

def set_web_design(page_title="Home", 
                   page_icon="./logo_imgs/logo.png", 
                   title="🤖 LangGraph RAG Algorithms", 
                   caption="",
                   logo_path = './logo_imgs/logo.png'):

    st.logo(image=logo_path, size='large', icon_image=logo_path)
    
    st.set_page_config(page_title=page_title, page_icon=page_icon)
    st.title(title)
    st.caption(caption)