from pathlib import Path

from finrag.common.exceptions import BadRequestError

_EXT = {
    ".txt": "text",
    ".md": "markdown",
    ".markdown": "markdown",
    ".pdf": "pdf",
}


def detect_doc_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in _EXT:
        raise BadRequestError(
            f"Unsupported file type: {ext!r}. Supported: {', '.join(sorted(_EXT))}"
        )
    return _EXT[ext]


def document_to_text(path: Path) -> tuple[str, dict]:
    ext = path.suffix.lower()
    if ext in (".txt", ".md", ".markdown"):
        text = path.read_text(encoding="utf-8", errors="replace")
        return text, {"page": 1}
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        parts: list[str] = []
        for i, page in enumerate(reader.pages, start=1):
            t = page.extract_text() or ""
            parts.append(f"<!-- page {i} -->\n{t}")
        return "\n\n".join(parts), {"pages": len(reader.pages)}
    raise BadRequestError(f"Cannot parse: {ext}")
