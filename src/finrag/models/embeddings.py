from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from finrag.common.config import Settings, get_settings


class _MockEmbeddings(Embeddings):
    """Hash-based pseudo embeddings for tests (768-d)."""

    def _vec(self, text: str) -> list[float]:
        import hashlib

        h = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
        return [((h[i % len(h)] + i) % 17) / 16.0 for i in range(384)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    s = settings or get_settings()
    if s.dev_mock_llm:
        return _MockEmbeddings()
    if not s.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set, or set DEV_MOCK_LLM=true for local tests.")
    kwargs: dict = {"model": s.embedding_model, "api_key": s.openai_api_key}
    if s.openai_base_url:
        kwargs["base_url"] = s.openai_base_url
    return OpenAIEmbeddings(**kwargs)
