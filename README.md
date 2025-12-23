# 1. Introduction

<p align="center"><img src="https://github.com/user-attachments/assets/cdceed9f-75c0-41a5-b728-b1c64485e48f"" width="30%" height="30%"></p>

## 프로젝트 소개
이 저장소는 최신 RAG 관련 연구 논문들의 핵심 아이디어와 아키텍처를 실용적으로 구현한 코드를 제공합니다. 각 구현은 논문의 세부 사항을 완벽히 재현하기보다는, 실무에서 활용 가능한 형태로 주요 개념과 핵심 메커니즘을 구현하는 데 중점을 둡니다.

LangGraph의 그래프 기반 워크플로우와 다양한 LLM API를 결합하여, 복잡한 RAG 파이프라인을 모듈화되고 확장 가능한 형태로 구성합니다. 이를 통해 각 RAG 기법의 작동 원리를 이해하고, 실제 프로젝트에 적용할 수 있는 실용적인 코드베이스를 제공합니다.

이 저장소는 OpenAI에서 제공하는 LLM API를 효과적으로 활용하여 RAG 시스템을 구축하는 방법에 집중합니다.

## 주요 특징
- 인터랙티브 데모 : 모든 RAG 파이프라인은 Streamlit 기반의 웹 인터페이스로 제공되어, 코드를 직접 실행하고 결과를 즉시 확인할 수 있습니다.
- 실용적 구현 : 논문의 핵심 아이디어를 실무 환경에서 바로 적용 가능한 형태로 구현
- 모듈화된 구조 : LangGraph를 활용한 체계적이고 확장 가능한 파이프라인 설계

## 구현 범위
- 포함 사항 : 검색 전략, 문서 처리, 쿼리 변환, 응답 생성 등 RAG 시스템의 핵심 컴포넌트 구현
- 제외 사항 : SFT(Supervised Fine-Tuning), Preference Alignment(RLHF, DPO 등) 등 LLM 모델 자체를 학습하거나 Fine-Tuning하는 방법론은 다루지 않습니다.

## 프로젝트 구조
```bash
├── core/                    # RAG 파이프라인 핵심 구현 코드
├── pages/                   # Streamlit 실행을 위한 페이지 코드
├── sys_prompt_hub/          # LLM 실행을 위한 시스템 프롬프트 모음
├── user_uploaded_files/     # 사용자 업로드 파일 저장 디렉토리
└── utils/                   # RAG 실행을 위한 유틸리티 함수
```
python 3.10 버전을 사용합니다.

# 2. Streamlit 홈페이지 구조

<p align="center"><img src="https://github.com/user-attachments/assets/fa0b9c9b-d2cd-49a8-87f2-09e7f1fa46ce"" width="100%" height="100%"></p>

## streamlit 실행 명령어
먼저, cmd 창에서 해당 프로젝트 경로로 이동 후, 아래와 같은 명령어를 입력합니다.

```bash
streamlit run Home.py --server.port [port_num]
```

## 주요 파라미터 설정
- LLM 모델 선택 : gpt4.1, gpt-4.1-mini, gpt-4.1-nano 모델 중 하나를 선택합니다. 최종 답변을 수행하는 LLM 모델입니다.
- temperature : 0~1 사이의 실수를 선택하는 것이며, 높을수록 좀 더 창의적인 답변을 얻을 수 있습니다.
- top-k : 검색되는 문서의 수를 의미하며, 1~10 사이의 값으로 선택할 수 있습니다.
- 파일 업로드 : RAG 실행을 위한 문서를 업로드 합니다. .docx, .pdf, .hwpx, .txt, .md 파일을 지원합니다.
- OpenAI API Key : LLM를 API로 사용하기 위한 OpenAI의 api key를 입력합니다. OpenAI Playground에서 회원가입 후 발급받을 수 있습니다.
- Tavily API Key : 특정 RAG의 경우, 웹 검색을 필요로 합니다. Tavliy 회원가입 후 api key를 발급받을 수 있습니다.
- Neo4j 관련 : Graph RAG의 경우, graphdb 생성을 필요로 합니다. neo4j 회원가입 후 instance 만드시면 텍스트 파일로 username, url, password를 발급받을 수 있습니다.

# 3. RAG 파이프라인 구현 현황

|이름|논문 링크|구현 완료 여부|
|------|---|---|
|Naive RAG|없음|완료|
|Long RAG|https://arxiv.org/abs/2410.18050|완료|
|Corrective RAG|https://arxiv.org/abs/2401.15884|완료|
|Self RAG|https://arxiv.org/abs/2310.11511|완료|
|MARAG|https://arxiv.org/abs/2505.20096|완료|
|MainRAG|https://arxiv.org/abs/2501.00332|완료|
|MadamRAG|https://arxiv.org/abs/2504.13079|완료|
|Naive GraphRAG|없음|완료|
|Self Route|https://arxiv.org/abs/2505.20664|완료|
|RECOMP|https://arxiv.org/abs/2310.04408|완료|
|WritingPath|https://arxiv.org/abs/2404.13919|완료|

이 프로젝트는 비정기적으로 (최대한 자주) 업데이트 될 예정입니다. 궁긍하신 사항은 깃허브 이슈나 제 개인 메일 계정 cjw94103@gmail.com으로 연락주시길 바랍니다.
감사합니다.
