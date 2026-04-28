from __future__ import annotations

from operator import itemgetter
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel

from finrag.common.config import get_settings
from finrag.models.llm import get_chat_model


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是金融知识助手。仅根据用户提供的「参考资料」作答；"
            "若资料不足以回答，请明确说明「资料中未找到相关信息」，不要编造数据与监管条款原文。"
            "如有多处来源，在相关句子后用简短角标式来源提示（例如「来源：文档名，约第N页」）。",
        ),
        (
            "human",
            "参考资料：\n{context}\n\n-----\n问题：{question}",
        ),
    ]
)


def _format_docs(docs: list) -> str:
    parts: list[str] = []
    for d in docs:
        m = d.metadata
        head = f"[{m.get('source', '?')}|p.{m.get('page', '?')}|{m.get('chunk_id', '')}]"
        parts.append(f"{head}\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def build_rag_chain(retriever: BaseRetriever) -> Runnable:
    s = get_settings()
    llm = get_chat_model(s)

    def retrieve_docs(q: str) -> str:
        docs = retriever.invoke(q) if hasattr(retriever, "invoke") else retriever.get_relevant_documents(q)
        return _format_docs(docs)

    retrieve_runnable: Runnable = RunnableLambda(retrieve_docs)

    return (
        RunnableParallel(
            {
                "context": itemgetter("question") | retrieve_runnable,
                "question": itemgetter("question"),
            }
        )
        | PROMPT
        | llm
        | StrOutputParser()
    )


def build_streaming_rag(retriever: BaseRetriever) -> Runnable:
    s = get_settings()
    llm = get_chat_model(s)

    def retrieve_docs(q: str) -> str:
        docs = retriever.invoke(q) if hasattr(retriever, "invoke") else retriever.get_relevant_documents(q)
        return _format_docs(docs)

    retrieve_runnable: Runnable = RunnableLambda(retrieve_docs)
    return (
        RunnableParallel(
            {
                "context": itemgetter("question") | retrieve_runnable,
                "question": itemgetter("question"),
            }
        )
        | PROMPT
        | llm
    )


def get_retrieved_citations(retriever: BaseRetriever, question: str) -> list[dict[str, Any]]:
    docs = retriever.invoke(question) if hasattr(retriever, "invoke") else retriever.get_relevant_documents(
        question
    )
    return [
        {
            "chunk_id": d.metadata.get("chunk_id"),
            "source": d.metadata.get("source"),
            "page": d.metadata.get("page"),
            "snippet": d.page_content[:500],
        }
        for d in docs
    ]
