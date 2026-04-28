from __future__ import annotations

from pydantic import BaseModel, Field


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = ""


class KnowledgeBaseOut(BaseModel):
    id: str
    name: str
    description: str
    created_at: str
    index_version: int


class DocumentOut(BaseModel):
    id: str
    filename: str
    status: str
    created_at: str
    error_message: str | None = None


class JobOut(BaseModel):
    id: str
    kind: str
    status: str
    kb_id: str
    document_id: str | None
    message: str | None
    created_at: str
    updated_at: str


class CitationOut(BaseModel):
    chunk_id: str | None = None
    source: str | None = None
    page: int | str | None = None
    snippet: str = ""


class ChatMessageRef(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    knowledge_base_id: str
    message: str = Field(min_length=1, max_length=12000)
    history: list[ChatMessageRef] = Field(default_factory=list)


class ChatResponse(BaseModel):
    id: str
    answer: str
    citations: list[CitationOut]
    knowledge_base_id: str
    disclaimer: str = "本回答由 AI 根据检索材料生成，不构成投资建议、法律或审计意见。请以原始文档与专业顾问意见为准。"


class HealthOut(BaseModel):
    status: str
    service: str = "finance-rag-ft"
