import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

from finrag.common.config import get_settings

UTC = timezone.utc


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class KnowledgeBaseRow:
    id: str
    name: str
    description: str
    created_at: str
    index_version: int


@dataclass
class DocumentRow:
    id: str
    kb_id: str
    filename: str
    path: str
    doc_type: str
    status: str
    created_at: str
    error_message: str | None


@dataclass
class JobRow:
    id: str
    kind: str
    status: str
    kb_id: str
    document_id: str | None
    message: str | None
    created_at: str
    updated_at: str


class Database:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path or get_settings().sqlite_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS knowledge_bases (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    index_version INTEGER NOT NULL DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    kb_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    path TEXT NOT NULL,
                    doc_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id)
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    kb_id TEXT NOT NULL,
                    document_id TEXT,
                    message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_log (
                    id TEXT PRIMARY KEY,
                    event TEXT NOT NULL,
                    detail TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )

    def create_knowledge_base(self, name: str, description: str = "") -> KnowledgeBaseRow:
        kb_id = str(uuid.uuid4())
        with self._conn() as c, self._lock:
            c.execute(
                "INSERT INTO knowledge_bases (id, name, description, created_at) VALUES (?, ?, ?, ?)",
                (kb_id, name, description, _now_iso()),
            )
        return self.get_knowledge_base(kb_id)  # type: ignore[return-value]

    def get_knowledge_base(self, kb_id: str) -> KnowledgeBaseRow | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM knowledge_bases WHERE id = ?", (kb_id,)).fetchone()
        if not r:
            return None
        return KnowledgeBaseRow(
            id=r["id"],
            name=r["name"],
            description=r["description"],
            created_at=r["created_at"],
            index_version=r["index_version"],
        )

    def list_knowledge_bases(self) -> list[KnowledgeBaseRow]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM knowledge_bases ORDER BY created_at DESC").fetchall()
        return [
            KnowledgeBaseRow(
                id=r["id"],
                name=r["name"],
                description=r["description"],
                created_at=r["created_at"],
                index_version=r["index_version"],
            )
            for r in rows
        ]

    def create_document(
        self, kb_id: str, filename: str, rel_path: str, doc_type: str
    ) -> DocumentRow:
        doc_id = str(uuid.uuid4())
        with self._conn() as c, self._lock:
            c.execute(
                """INSERT INTO documents (id, kb_id, filename, path, doc_type, status, created_at)
                VALUES (?, ?, ?, ?, ?, 'pending', ?)""",
                (doc_id, kb_id, filename, rel_path, doc_type, _now_iso()),
            )
        return self.get_document(doc_id)  # type: ignore[return-value]

    def get_document(self, doc_id: str) -> DocumentRow | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if not r:
            return None
        return DocumentRow(
            id=r["id"],
            kb_id=r["kb_id"],
            filename=r["filename"],
            path=r["path"],
            doc_type=r["doc_type"],
            status=r["status"],
            created_at=r["created_at"],
            error_message=r["error_message"],
        )

    def update_document_status(
        self, doc_id: str, status: str, error: str | None = None
    ) -> None:
        with self._conn() as c, self._lock:
            c.execute(
                "UPDATE documents SET status = ?, error_message = ? WHERE id = ?",
                (status, error, doc_id),
            )

    def list_documents(self, kb_id: str) -> list[DocumentRow]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM documents WHERE kb_id = ? ORDER BY created_at DESC", (kb_id,)
            ).fetchall()
        return [
            DocumentRow(
                id=r["id"],
                kb_id=r["kb_id"],
                filename=r["filename"],
                path=r["path"],
                doc_type=r["doc_type"],
                status=r["status"],
                created_at=r["created_at"],
                error_message=r["error_message"],
            )
            for r in rows
        ]

    def create_job(
        self, kind: str, kb_id: str, document_id: str | None = None, message: str | None = None
    ) -> JobRow:
        job_id = str(uuid.uuid4())
        t = _now_iso()
        with self._conn() as c, self._lock:
            c.execute(
                """INSERT INTO jobs (id, kind, status, kb_id, document_id, message, created_at, updated_at)
                VALUES (?, ?, 'queued', ?, ?, ?, ?, ?)""",
                (job_id, kind, kb_id, document_id, message, t, t),
            )
        return self.get_job(job_id)  # type: ignore[return-value]

    def get_job(self, job_id: str) -> JobRow | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not r:
            return None
        return JobRow(
            id=r["id"],
            kind=r["kind"],
            status=r["status"],
            kb_id=r["kb_id"],
            document_id=r["document_id"],
            message=r["message"],
            created_at=r["created_at"],
            updated_at=r["updated_at"],
        )

    def update_job(self, job_id: str, status: str, message: str | None = None) -> None:
        with self._conn() as c, self._lock:
            c.execute(
                "UPDATE jobs SET status = ?, message = ?, updated_at = ? WHERE id = ?",
                (status, message, _now_iso(), job_id),
            )

    def audit(self, event: str, detail: str | None = None) -> None:
        eid = str(uuid.uuid4())
        with self._conn() as c, self._lock:
            c.execute(
                "INSERT INTO audit_log (id, event, detail, created_at) VALUES (?, ?, ?, ?)",
                (eid, event, detail, _now_iso()),
            )


_db: Database | None = None


def get_db() -> Database:
    global _db
    if _db is None:
        _db = Database()
    return _db
