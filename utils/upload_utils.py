import os
import streamlit as st

## 업로드한 파일을 저장하기 위한 캐시 디렉토리 생성
def make_rag_cache_dir(directory):
    if not os.path.exists(directory): # 해당 이름의 폴더가 존재하는지 여부 확인
        os.makedirs(directory)

def save_uploaded_file(directory, file):
    filepath = os.path.join(directory, file.name)
    with open(filepath, 'wb') as f:
        f.write(file.getbuffer()) # 해당 내용은 Buffer로 작성하겠다.

    return filepath

## 업로드한 파일을 세이브하는 함수
def save_cache_files(directory, uploaded_files):
    filepaths = []
    for i, file in enumerate(uploaded_files):
        if i==0:
            make_rag_cache_dir(directory)
            filepath = save_uploaded_file(directory, file)
            filepaths.append(filepath)
            # st.success("{} 업로드 성공".format(filepath))
        else:
            filepath = save_uploaded_file(directory, file)
            filepaths.append(filepath)
            # st.success("{} 업로드 성공".format(filepath))

    return filepaths