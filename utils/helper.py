import os
import openai
import pickle
import faiss

from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from utils.document_loader import process_middle_docx, process_fast_hwpx, load_pdf_documents, load_txt_as_documents, load_md_as_documents

# OpenAI API Key 설정
def openai_api_setting(api_key=None):
    
    os.environ["OPENAI_API_KEY"] = api_key
    openai.api_key = api_key

# Tavily API Key 설정
def tavily_api_setting(api_key=None):
    os.environ["TAVILY_API_KEY"] = api_key

# 텍스트 파일에서 문자열을 읽어오는 함수
def load_string_from_txt(filename: str) -> str:
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

# 유저가 업로드한 파일에서 VectorDB 만들기
# pdf, docx, hwpx, md, txt 포맷 지원
def make_uploadfile_to_retriever(filepaths,
                                embedding_function=None,
                                embedding_dim=3072,
                                parent_chunk_size=1000,
                                parent_chunk_overlap=100,
                                child_chunk_size=200,
                                child_chunk_overlap=20,
                                top_k=3):
    
    vectorstore = FAISS(embedding_function=embedding_function,
                        index=faiss.IndexFlatL2(embedding_dim),
                        docstore=InMemoryDocstore({}),
                        index_to_docstore_id={})

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=parent_chunk_overlap)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_chunk_size, chunk_overlap=child_chunk_overlap)

    parent_store = InMemoryStore()
    user_retriever = ParentDocumentRetriever(vectorstore=vectorstore, 
                                             docstore=parent_store, 
                                             parent_splitter=parent_splitter, 
                                             child_splitter=child_splitter, 
                                             search_kwargs={"k": top_k}, 
                                             search_type="similarity")

    docs = []
    for path in filepaths:
        if ".docx" in path:
            docs.extend(process_middle_docx(path))
        elif ".hwpx" in path:
            docs.extend(process_fast_hwpx(path))
        elif ".pdf" in path:
            docs.extend(load_pdf_documents(path))
        elif ".txt" in path:
            docs.extend(load_txt_as_documents(path))
        elif ".md" in path:
            docs.extend(load_md_as_documents(path))
            
    user_retriever.add_documents(docs)
    
    return user_retriever

## dense retriever
def make_uploadfile_to_dense_retriever(filepaths,
                                      embedding_function=None,
                                      chunk_size=1000,
                                      chunk_overlap=100,
                                      top_k=3):
    # document 생성
    docs = []
    for path in filepaths:
        if ".docx" in path:
            docs.extend(process_middle_docx(path))
        elif ".hwpx" in path:
            docs.extend(process_fast_hwpx(path))
        elif ".pdf" in path:
            docs.extend(load_pdf_documents(path))
        elif ".txt" in path:
            docs.extend(load_txt_as_documents(path))
        elif ".md" in path:
            docs.extend(load_md_as_documents(path))

    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(docs)

    # vectorstore 생성
    vectorstore = FAISS.from_documents(documents=split_docs, embedding=embedding_function)

    # dense retriever 생성
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})

    return retriever

def save_parent_retriever(retriever, save_dir):
    """ParentDocumentRetriever를 로컬에 저장"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 1. FAISS vectorstore 저장
    retriever.vectorstore.save_local(str(save_path / "vectorstore"))
    
    # 2. Parent docstore 저장 (InMemoryStore)
    with open(save_path / "parent_store.pkl", "wb") as f:
        pickle.dump(retriever.docstore.store, f)
    
    # 3. 설정 정보 저장 (재구성에 필요)
    config = {
        "parent_chunk_size": retriever.parent_splitter._chunk_size,
        "parent_chunk_overlap": retriever.parent_splitter._chunk_overlap,
        "child_chunk_size": retriever.child_splitter._chunk_size,
        "child_chunk_overlap": retriever.child_splitter._chunk_overlap,
        "search_kwargs": retriever.search_kwargs,
        "search_type": retriever.search_type
    }
    with open(save_path / "config.pkl", "wb") as f:
        pickle.dump(config, f)
    
    print(f"Retriever saved to {save_dir}")

def load_parent_retriever(save_dir, embedding_function):
    """저장된 ParentDocumentRetriever 불러오기"""
    save_path = Path(save_dir)
    
    # 1. FAISS vectorstore 불러오기
    vectorstore = FAISS.load_local(
        str(save_path / "vectorstore"),
        embedding_function,
        allow_dangerous_deserialization=True
    )
    
    # 2. Parent docstore 불러오기
    with open(save_path / "parent_store.pkl", "rb") as f:
        store_dict = pickle.load(f)
    parent_store = InMemoryStore()
    parent_store.store = store_dict
    
    # 3. 설정 불러오기
    with open(save_path / "config.pkl", "rb") as f:
        config = pickle.load(f)
    
    # 4. Retriever 재구성
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["parent_chunk_size"],
        chunk_overlap=config["parent_chunk_overlap"]
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["child_chunk_size"],
        chunk_overlap=config["child_chunk_overlap"]
    )
    
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=parent_store,
        parent_splitter=parent_splitter,
        child_splitter=child_splitter,
        search_kwargs=config["search_kwargs"],
        search_type=config["search_type"]
    )
    
    return retriever

def extract_model_provider(model_name: str) -> str:

    if "gpt" in model_name:
        return "openai"
    elif "claude" in model_name:
        return "anthropic"
    # 암묵적 None 반환 (다른 제공자는 미지원)


def model_name_to_max_tokens(model_name: str) -> int:
    
    if "gpt-4o" in model_name:
        return 14000
    elif "claude-3-7-sonnet" in model_name:
        return 62000
    elif "claude-3-5" in model_name:
        return 4000
    elif "gpt-5" in model_name:
        return 62000
    elif "claude-sonnet-4" in model_name:
        return 62000
    elif "claude-opus-4" in model_name:
        return 30000
    elif "gpt-4.1" in model_name:
        return 28000
        