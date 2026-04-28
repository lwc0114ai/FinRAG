import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from finrag.api.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentOut,
    HealthOut,
    JobOut,
    KnowledgeBaseCreate,
    KnowledgeBaseOut,
)
from finrag.app.chat import chat_once, stream_chat
from finrag.common.exceptions import BadRequestError, NotFoundError
from finrag.ingest.pipeline import run_index_job, save_uploaded_file
from finrag.persistence.db import Database, get_db
from finrag.security.auth import optional_api_key

router = APIRouter()


def get_database() -> Database:
    return get_db()


@router.get("/v1/health", response_model=HealthOut, tags=["health"])
def health() -> HealthOut:
    return HealthOut(status="ok")


@router.post(
    "/v1/knowledge_bases",
    response_model=KnowledgeBaseOut,
    tags=["knowledge_bases"],
    dependencies=[Depends(optional_api_key)],
)
def create_kb(
    body: KnowledgeBaseCreate,
    db: Database = Depends(get_database),
) -> KnowledgeBaseOut:
    row = db.create_knowledge_base(body.name, body.description)
    db.audit("kb_created", f"id={row.id} name={row.name!r}")
    return KnowledgeBaseOut(
        id=row.id,
        name=row.name,
        description=row.description,
        created_at=row.created_at,
        index_version=row.index_version,
    )


@router.get(
    "/v1/knowledge_bases",
    response_model=list[KnowledgeBaseOut],
    tags=["knowledge_bases"],
    dependencies=[Depends(optional_api_key)],
)
def list_kbs(db: Database = Depends(get_database)) -> list[KnowledgeBaseOut]:
    return [
        KnowledgeBaseOut(
            id=r.id,
            name=r.name,
            description=r.description,
            created_at=r.created_at,
            index_version=r.index_version,
        )
        for r in db.list_knowledge_bases()
    ]


@router.get(
    "/v1/knowledge_bases/{kb_id}",
    response_model=KnowledgeBaseOut,
    tags=["knowledge_bases"],
    dependencies=[Depends(optional_api_key)],
)
def get_kb(kb_id: str, db: Database = Depends(get_database)) -> KnowledgeBaseOut:
    r = db.get_knowledge_base(kb_id)
    if not r:
        raise NotFoundError("knowledge base not found")
    return KnowledgeBaseOut(
        id=r.id,
        name=r.name,
        description=r.description,
        created_at=r.created_at,
        index_version=r.index_version,
    )


@router.get(
    "/v1/knowledge_bases/{kb_id}/documents",
    response_model=list[DocumentOut],
    tags=["documents"],
    dependencies=[Depends(optional_api_key)],
)
def list_docs(kb_id: str, db: Database = Depends(get_database)) -> list[DocumentOut]:
    if not db.get_knowledge_base(kb_id):
        raise NotFoundError("knowledge base not found")
    return [
        DocumentOut(
            id=d.id, filename=d.filename, status=d.status, created_at=d.created_at, error_message=d.error_message
        )
        for d in db.list_documents(kb_id)
    ]


@router.post(
    "/v1/knowledge_bases/{kb_id}/documents",
    tags=["documents"],
    dependencies=[Depends(optional_api_key)],
)
async def upload_document(
    kb_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Database = Depends(get_database),
) -> JSONResponse:
    if not file.filename:
        raise BadRequestError("missing filename")
    if not db.get_knowledge_base(kb_id):
        raise NotFoundError("knowledge base not found")
    data = await file.read()
    if not data:
        raise BadRequestError("empty file")
    try:
        doc_type, rel = save_uploaded_file(kb_id, file.filename, data)
    except Exception as e:  # noqa: BLE001
        raise BadRequestError(str(e)) from e
    doc = db.create_document(kb_id, Path(file.filename).name, rel, doc_type)
    job = db.create_job("index_document", kb_id, document_id=doc.id, message=doc.id)
    db.audit("document_uploaded", f"kb_id={kb_id} doc_id={doc.id} job_id={job.id}")

    async def _run() -> None:
        await asyncio.to_thread(run_index_job, db, job.id, kb_id, doc.id)

    background_tasks.add_task(_run)
    return JSONResponse(
        status_code=202,
        content={
            "document_id": doc.id,
            "job_id": job.id,
            "status": "queued",
        },
    )


@router.get(
    "/v1/jobs/{job_id}",
    response_model=JobOut,
    tags=["jobs"],
    dependencies=[Depends(optional_api_key)],
)
def get_job(job_id: str, db: Database = Depends(get_database)) -> JobOut:
    j = db.get_job(job_id)
    if not j:
        raise NotFoundError("job not found")
    return JobOut(
        id=j.id,
        kind=j.kind,
        status=j.status,
        kb_id=j.kb_id,
        document_id=j.document_id,
        message=j.message,
        created_at=j.created_at,
        updated_at=j.updated_at,
    )


@router.post(
    "/v1/chat",
    response_model=ChatResponse,
    tags=["chat"],
    dependencies=[Depends(optional_api_key)],
)
def chat(
    body: ChatRequest,
    db: Database = Depends(get_database),
) -> ChatResponse:
    if not db.get_knowledge_base(body.knowledge_base_id):
        raise NotFoundError("knowledge base not found")
    hist = [h.model_dump() for h in body.history]
    out = chat_once(body.knowledge_base_id, body.message, hist)
    return ChatResponse(
        id=out["id"],
        answer=out["answer"],
        citations=out["citations"],
        knowledge_base_id=out["knowledge_base_id"],
    )


@router.post(
    "/v1/chat/stream",
    tags=["chat"],
    dependencies=[Depends(optional_api_key)],
)
def chat_stream(
    body: ChatRequest,
    db: Database = Depends(get_database),
) -> StreamingResponse:
    if not db.get_knowledge_base(body.knowledge_base_id):
        raise NotFoundError("knowledge base not found")
    hist = [h.model_dump() for h in body.history]

    def _iter():
        it, cits = stream_chat(body.knowledge_base_id, body.message, hist)
        for part in it:
            yield part
        if cits:
            yield f"\n\n__CITATIONS__={json.dumps(cits, ensure_ascii=False)}\n"

    return StreamingResponse(_iter(), media_type="text/plain; charset=utf-8")
