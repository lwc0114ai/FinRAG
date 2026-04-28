import os
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("DEV_MOCK_LLM", "true")


@pytest.fixture
def test_project_root() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        root = Path(td)
        (root / "data").mkdir()
        (root / "data" / "chroma").mkdir()
        (root / "data" / "storage").mkdir()
        (root / "data" / "lora_out").mkdir()
        os.environ["SQLITE_PATH"] = str(root / "data" / "finrag.db")
        os.environ["CHROMA_PERSIST_DIR"] = str(root / "data" / "chroma")
        os.environ["FILE_STORAGE_DIR"] = str(root / "data" / "storage")
        os.environ["DATA_DIR"] = str(root / "data")
        os.environ["DEV_MOCK_LLM"] = "true"
        from finrag.common import config
        import finrag.persistence.db as dbmod
        import finrag.rag.retrieve.retriever as rret

        config.get_settings.cache_clear()  # type: ignore[attr-defined]
        dbmod._db = None
        rret._cache.clear()
        yield root
        config.get_settings.cache_clear()
        dbmod._db = None
        rret._cache.clear()


@pytest.fixture
def client(test_project_root: Path) -> Generator[TestClient, None, None]:  # noqa: ARG001
    from finrag.api.main import app

    with TestClient(app) as c:
        yield c
