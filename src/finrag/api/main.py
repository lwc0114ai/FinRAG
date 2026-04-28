import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from finrag import __version__
from finrag.api.routes import router
from finrag.common.exceptions import FinRagError

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        from finrag.common.config import get_settings

        s = get_settings()
        s.data_dir  # side effect: create dirs
        from finrag.persistence.db import get_db

        get_db()  # init schema
    except Exception:  # noqa: BLE001
        log.exception("Startup failed")
    yield


app = FastAPI(
    title="金融 RAG + 微调 API",
    version=__version__,
    lifespan=lifespan,
    description="文档入库、向量化与 LangChain 检索问答。详见 /docs 。",
)
app.include_router(router, prefix="")


@app.exception_handler(FinRagError)
def finrag_error_handler(_request: Request, exc: FinRagError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"code": exc.code, "message": exc.message},
    )
