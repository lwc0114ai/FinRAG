import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from finrag.common.config import get_settings

_PAGE_MARK = re.compile(r"<!--\s*page\s+(\d+)\s*-->")


def _page_for_offset(text: str, offset: int) -> int:
    before = text[:offset]
    pages = [int(m.group(1)) for m in _PAGE_MARK.finditer(before)]
    return pages[-1] if pages else 1


def chunk_documents(
    text: str,
    doc_id: str,
    filename: str,
    kb_id: str,
) -> list[Document]:
    s = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
        separators=["\n\n", "\n", "。", "；", " ", ""],
    )
    sub_chunks = splitter.create_documents([text])
    out: list[Document] = []
    # Offsets: find chunk start in original text
    pos = 0
    for i, d in enumerate(sub_chunks):
        chunk_text = d.page_content
        start = text.find(chunk_text, pos) if chunk_text in text else text.find(chunk_text[:200], pos)
        if start < 0:
            start = pos
        page = _page_for_offset(text, start)
        pos = start + max(len(chunk_text), 1)
        out.append(
            Document(
                page_content=chunk_text,
                metadata={
                    "chunk_id": f"{doc_id}_c{i}",
                    "doc_id": doc_id,
                    "kb_id": kb_id,
                    "source": filename,
                    "page": page,
                },
            )
        )
    return out
