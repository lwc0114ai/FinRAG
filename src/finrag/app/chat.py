from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any

from finrag.common.exceptions import NotFoundError
from finrag.persistence.db import get_db
from finrag.rag.chains.rag_chain import build_rag_chain, build_streaming_rag, get_retrieved_citations
from finrag.rag.retrieve import get_retriever


def _merge_question(history: list[dict[str, str]] | None, message: str) -> str:
    if not history:
        return message
    lines: list[str] = []
    for m in history[-6:]:
        r = m.get("role", "user")
        c = m.get("content", "")
        if r in ("user", "assistant", "system") and c:
            lines.append(f"{r}: {c}")
    lines.append(f"user: {message}")
    return "\n".join(lines)


def chat_once(
    kb_id: str, message: str, history: list[dict[str, str]] | None = None
) -> dict[str, Any]:
    db = get_db()
    if not db.get_knowledge_base(kb_id):
        raise NotFoundError("knowledge base not found")
    retriever = get_retriever(kb_id)
    chain = build_rag_chain(retriever)
    q = _merge_question(history, message)
    text = chain.invoke({"question": q})
    cits = get_retrieved_citations(retriever, q)
    return {
        "id": str(uuid.uuid4()),
        "answer": text,
        "citations": cits,
        "knowledge_base_id": kb_id,
    }


def stream_chat(
    kb_id: str, message: str, history: list[dict[str, str]] | None = None
) -> tuple[Iterator[str], list[dict[str, Any]]]:
    if not get_db().get_knowledge_base(kb_id):
        raise NotFoundError("knowledge base not found")
    retriever = get_retriever(kb_id)
    chain = build_streaming_rag(retriever)
    q = _merge_question(history, message)
    cits = get_retrieved_citations(retriever, q)

    def gen() -> Iterator[str]:
        for chunk in chain.stream({"question": q}):
            if hasattr(chunk, "content") and chunk.content:
                t = chunk.content
                if isinstance(t, str):
                    yield t
                else:
                    yield str(t)
            else:
                yield str(chunk)

    return gen(), cits
