from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever
from pydantic import ConfigDict, Field

from finrag.common.config import get_settings
from finrag.rag.index.store import get_chroma, load_all_derived_chunks

_cache: dict[str, BaseRetriever] = {}


def invalidate_kb_cache(kb_id: str) -> None:
    _cache.pop(kb_id, None)


def rrf_fuse_retrievers(
    retrievers: Sequence[BaseRetriever],
    query: str,
    k: int,
    weights: Sequence[float] | None = None,
    c: int = 60,
) -> list[Document]:
    if not retrievers:
        return []
    w = list(weights) if weights else [1.0 / len(retrievers)] * len(retrievers)
    key_scores: dict[str, float] = defaultdict(float)
    key_doc: dict[str, Document] = {}
    for wi, ret in zip(w, retrievers, strict=False):
        docs: list[Document] = list(ret.invoke(query))  # type: ignore[union-attr]
        for rank, d in enumerate(docs):
            md = d.metadata or {}
            cid = md.get("chunk_id") or d.page_content[:200]
            s = str(cid)
            key_scores[s] += float(wi) * (1.0 / (c + rank + 1))
            if s not in key_doc:
                key_doc[s] = d
    top = sorted(key_scores.items(), key=lambda x: -x[1])[:k]
    return [key_doc[i] for i, _ in top]


class RRFEnsembleRetriever(BaseRetriever):
    """
    多路检索 RRF 融合。LangChain 新版已移除 `langchain.retrievers.ensemble` 时使用本类。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    rrf_retrievers: list[BaseRetriever] = Field(..., min_length=1)
    rrf_weights: list[float] = Field(default_factory=lambda: [0.5, 0.5])
    final_k: int = Field(default=5, ge=1, le=100)
    c_const: int = Field(default=60, ge=1, description="RRF rank constant k")

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,  # noqa: ARG002
    ) -> list[Document]:
        return rrf_fuse_retrievers(
            self.rrf_retrievers,
            query,
            k=self.final_k,
            weights=self.rrf_weights,
            c=self.c_const,
        )


def get_retriever(kb_id: str) -> BaseRetriever:
    s = get_settings()
    if kb_id in _cache:
        return _cache[kb_id]

    vs = get_chroma(kb_id)
    if s.use_mmr and not s.use_hybrid:
        retriever: BaseRetriever = vs.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": s.retrieval_k,
                "fetch_k": s.retrieval_k * 2,
                "lambda_mult": s.mmr_lambda,
            },
        )
    else:
        retriever = vs.as_retriever(search_kwargs={"k": s.retrieval_k})

    if s.use_hybrid:
        docs = load_all_derived_chunks(kb_id)
        if docs:
            try:
                bm25 = BM25Retriever.from_documents(docs, k1=1.5, b=0.75)  # type: ignore[call-arg]
            except TypeError:
                bm25 = BM25Retriever.from_documents(docs)  # type: ignore[call-arg]
            bm25.k = s.retrieval_k
            retriever = RRFEnsembleRetriever(
                rrf_retrievers=[retriever, bm25],  # type: ignore[assignment, arg-type]
                rrf_weights=[0.5, 0.5],
                final_k=s.retrieval_k,
            )

    _cache[kb_id] = retriever
    return retriever
