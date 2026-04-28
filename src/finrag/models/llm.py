from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from finrag.common.config import Settings, get_settings


def get_chat_model(settings: Settings | None = None) -> BaseChatModel:
    s = settings or get_settings()
    if s.dev_mock_llm:
        return FakeListChatModel(
            responses=[
                "[mock-llm] Based on retrieved context (dev placeholder). Set OPENAI_API_KEY for real answers.",
            ]
        )
    if not s.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set. Copy .env.example to .env or set DEV_MOCK_LLM=true.")
    kwargs: dict = {
        "model": s.llm_model,
        "api_key": s.openai_api_key,
    }
    if s.openai_base_url:
        kwargs["base_url"] = s.openai_base_url
    return ChatOpenAI(**kwargs, temperature=0.0)
