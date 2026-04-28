from pathlib import Path

from finrag.common.config import get_settings
from finrag.persistence.db import Database, get_db
from finrag.rag.index import add_documents_to_index, persist_chunk_lines
from finrag.rag.parse import chunk_documents, document_to_text, detect_doc_type
from finrag.rag.retrieve import invalidate_kb_cache


def run_index_job(db: Database, job_id: str, kb_id: str, document_id: str) -> None:
    s = get_settings()
    doc = db.get_document(document_id)
    if not doc:
        db.update_job(job_id, "failed", "document not found")
        return
    root = s.file_storage_dir
    try:
        db.update_job(job_id, "running", None)
        db.update_document_status(document_id, "parsing")
        path = (root / doc.path).resolve()
        if not path.is_file() or not str(path).startswith(str(root.resolve())):
            raise ValueError("invalid storage path")
        text, _extra = document_to_text(path)
        db.update_document_status(document_id, "indexing")
        chunks = chunk_documents(text, document_id, doc.filename, kb_id)
        if not chunks:
            raise ValueError("no text extracted")
        add_documents_to_index(kb_id, chunks)
        persist_chunk_lines(kb_id, chunks)
        invalidate_kb_cache(kb_id)
        db.update_document_status(document_id, "indexed", None)
        db.update_job(job_id, "done", f"chunks={len(chunks)}")
        db.audit("document_indexed", f"kb_id={kb_id} doc_id={document_id} chunks={len(chunks)}")
    except Exception as e:  # noqa: BLE001
        db.update_document_status(document_id, "failed", str(e)[:2000])
        db.update_job(job_id, "failed", str(e)[:2000])
        db.audit("document_index_failed", f"doc_id={document_id} err={e!s}")


def save_uploaded_file(
    storage_rel_dir: str, filename: str, data: bytes
) -> tuple[str, str]:
    """Return (doc_type, relative path under file_storage)."""
    doc_type = detect_doc_type(filename)
    base = get_settings().file_storage_dir / storage_rel_dir
    base.mkdir(parents=True, exist_ok=True)
    safe = Path(filename).name
    path = base / safe
    path.write_bytes(data)
    rel = f"{storage_rel_dir}/{safe}"
    return doc_type, rel
