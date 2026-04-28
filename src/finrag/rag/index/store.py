import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from finrag.common.config import get_settings
from finrag.models.embeddings import get_embeddings


def kb_collection_name(kb_id: str) -> str:
    safe = "".join(c for c in kb_id if c.isalnum())[:64]
    return f"kb_{safe or 'x'}"


def get_chroma(kb_id: str) -> Chroma:
    s = get_settings()
    return Chroma(
        collection_name=kb_collection_name(kb_id),
        persist_directory=str(s.chroma_persist_dir),
        embedding_function=get_embeddings(s),
    )


def add_documents_to_index(kb_id: str, documents: list[Document]) -> list[str]:
    vs = get_chroma(kb_id)
    ids = [d.metadata.get("chunk_id", str(i)) for i, d in enumerate(documents)]
    vs.add_documents(documents, ids=ids)
    return ids


def derived_chunks_path(kb_id: str) -> Path:
    p = get_settings().data_dir / "derived" / kb_id
    p.mkdir(parents=True, exist_ok=True)
    return p / "all_chunks.jsonl"


def persist_chunk_lines(kb_id: str, documents: list[Document]) -> None:
    path = derived_chunks_path(kb_id)
    with path.open("a", encoding="utf-8") as f:
        for d in documents:
            f.write(
                json.dumps(
                    {"page_content": d.page_content, "metadata": d.metadata},
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_all_derived_chunks(kb_id: str) -> list[Document]:
    path = derived_chunks_path(kb_id)
    if not path.exists():
        return []
    out: list[Document] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        out.append(Document(page_content=obj["page_content"], metadata=obj["metadata"]))
    return out
