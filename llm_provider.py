import os
import warnings
from dotenv import load_dotenv

load_dotenv()

# provider별 필수 환경변수 정의
REQUIRED_ENVS = {
    "azure":     ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
                  "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_API_VERSION"],
    "openai":    ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
}


def _is_available(provider: str) -> bool:
    """provider에 필요한 환경변수가 모두 있는지 확인"""
    return all(os.getenv(k) for k in REQUIRED_ENVS.get(provider, []))


def _get_azure():
    from langchain_openai import AzureChatOpenAI
    return AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )


def _get_openai():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def _get_anthropic():
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )


PROVIDER_MAP = {
    "azure":     _get_azure,
    "openai":    _get_openai,
    "anthropic": _get_anthropic,
}


def get_llm(task: str = "default"):
    """
    task 이름에 맞는 LLM 인스턴스 반환.
    .env에서 LLM_PROVIDER_{TASK} 우선 조회,
    없으면 LLM_PROVIDER 기본값 사용.
    지정된 provider의 API 키가 없으면 기본값으로 fallback.

    사용 가능한 task:
      summarize  — 뉴스 요약
      sentiment  — 감성 판단 (LLM 사용 시)
      predict    — 주가 예측 판단
      report     — 리포트 생성
      default    — fallback
    """
    task_provider = os.getenv(f"LLM_PROVIDER_{task.upper()}")
    default_provider = os.getenv("LLM_PROVIDER", "azure")

    # task 전용 provider가 지정되어 있지만 API 키가 없으면 경고 후 fallback
    if task_provider and not _is_available(task_provider):
        warnings.warn(
            f"[llm_provider] '{task}' task에 지정된 provider '{task_provider}'의 "
            f"API 키가 없습니다. '{default_provider}'로 fallback합니다."
        )
        provider = default_provider
    else:
        provider = task_provider or default_provider

    if provider not in PROVIDER_MAP:
        raise ValueError(
            f"지원하지 않는 provider: '{provider}'\n"
            f"사용 가능: {list(PROVIDER_MAP.keys())}"
        )

    return PROVIDER_MAP[provider]()
