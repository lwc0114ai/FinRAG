from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")

    chunk_size: int = Field(default=800, alias="CHUNK_SIZE", ge=100, le=8000)
    chunk_overlap: int = Field(default=120, alias="CHUNK_OVERLAP", ge=0, le=2000)
    retrieval_k: int = Field(default=5, alias="RETRIEVAL_K", ge=1, le=50)
    rerank_top_n: int = Field(default=0, alias="RERANK_TOP_N", ge=0, le=20)
    use_hybrid: bool = Field(default=False, alias="USE_HYBRID")
    use_mmr: bool = Field(default=True, alias="USE_MMR")
    mmr_lambda: float = Field(default=0.5, alias="MMR_LAMBDA", ge=0.0, le=1.0)

    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")
    chroma_persist_dir: Path = Field(default=Path("./data/chroma"), alias="CHROMA_PERSIST_DIR")
    sqlite_path: Path = Field(default=Path("./data/finrag.db"), alias="SQLITE_PATH")
    file_storage_dir: Path = Field(default=Path("./data/storage"), alias="FILE_STORAGE_DIR")

    api_key: str = Field(default="", alias="API_KEY")
    jwt_secret: str = Field(default="dev-secret", alias="JWT_SECRET")
    session_ttl_minutes: int = Field(default=120, alias="SESSION_TTL_MINUTES", ge=1)

    train_data_path: Path = Field(default=Path("./training/data/train.jsonl"), alias="TRAIN_DATA_PATH")
    lora_output_dir: Path = Field(default=Path("./data/lora_out"), alias="LORA_OUTPUT_DIR")
    dev_mock_llm: bool = Field(default=False, alias="DEV_MOCK_LLM")

    @model_validator(mode="after")
    def ensure_dirs(self) -> "Settings":
        for p in (self.data_dir, self.chroma_persist_dir, self.file_storage_dir, self.lora_output_dir):
            p.mkdir(parents=True, exist_ok=True)
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
