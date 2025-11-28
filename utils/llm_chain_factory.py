import os

## langchain
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts.chat import (ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate)

from utils.helper import load_string_from_txt

def make_llm_chain(system_prompt_or_path, 
                   human_message="{question}",
                   temperature=0,
                   top_p=None,
                   max_tokens=None,
                   default_model: str = None,
                   default_provider: str = None,
                   configurable: bool = True):

    # init_chat_model 사용
    kwargs = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    # configurable_fields 설정
    if configurable:
        configurable_fields = ("model", "model_provider", "temperature", "max_tokens")
        config_prefix = ""  # prefix 없으면 config["configurable"]["model_provider"] 형식
    else:
        configurable_fields = None
        config_prefix = None

    # default_model / default_provider 값 있으면 넣고 없으면 None -> 런타임에 반드시 설정 필요
    llm = init_chat_model(
        model=default_model,
        model_provider=default_provider,
        **kwargs,
        configurable_fields=configurable_fields,
        config_prefix=config_prefix,
    )

    # 프롬프트 템플릿 준비
    if os.path.exists(system_prompt_or_path):
        system_message = load_string_from_txt(system_prompt_or_path)
    else:
        system_message = system_prompt_or_path

    system_prompt = SystemMessagePromptTemplate.from_template(system_message)
    human_prompt = HumanMessagePromptTemplate.from_template(human_message)
    llm_prompt_template = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    llm_chain = llm_prompt_template | llm | StrOutputParser()

    return llm_chain


def make_structed_llm_chain(
    system_prompt_or_path: str,
    human_message: str = "{question}",
    temperature: float = 0,
    top_p: float = None,
    max_tokens: int = None,
    default_model: str = None,
    default_provider: str = None,
    structed_class=None,
    configurable: bool = True):
    
    # LLM 초기화 파라미터
    init_kwargs = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    # Configurable 설정
    if configurable:
        configurable_fields = ("model", "model_provider", "temperature", "max_tokens")
        config_prefix = None
    else:
        configurable_fields = None
        config_prefix = None

    # LLM 초기화
    llm = init_chat_model(
        model=default_model,
        model_provider=default_provider,
        **init_kwargs,
        configurable_fields=configurable_fields,
        config_prefix=config_prefix,
    )

    # 구조화된 출력 설정
    if structed_class is not None:
        llm_structed = llm.with_structured_output(structed_class)
    else:
        llm_structed = llm

    # 프롬프트 구성
    if os.path.exists(system_prompt_or_path):
        system_message = load_string_from_txt(system_prompt_or_path)
    else:
        system_message = system_prompt_or_path

    system_prompt = SystemMessagePromptTemplate.from_template(system_message)
    human_prompt = HumanMessagePromptTemplate.from_template(human_message)
    llm_prompt_template = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    # 최종 체인: Prompt | LLM(with_structured_output)
    llm_chain = llm_prompt_template | llm_structed

    return llm_chain