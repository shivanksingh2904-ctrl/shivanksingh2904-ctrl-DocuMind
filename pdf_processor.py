"""Read PDFs and split them into overlapping chunks that remember their page number."""
from __future__ import annotations

import io
import re
from pypdf import PdfReader

MAX_PAGES = 300          # keeps the free host from running out of memory


def extract_pages(data: bytes) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...] for pages that contain text."""
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            raise ValueError("This PDF is password protected.")
    pages = []
    for number, page in enumerate(reader.pages[:MAX_PAGES], start=1):
        text = re.sub(r"[ \t]+", " ", page.extract_text() or "").strip()
        if text:
            pages.append((number, text))
    return pages


def chunk_text(text: str, words: int = 140, overlap: int = 30) -> list[str]:
    tokens = text.split()
    if not tokens:
        return []
    step = max(1, words - overlap)
    chunks = []
    for start in range(0, len(tokens), step):
        chunks.append(" ".join(tokens[start:start + words]))
        if start + words >= len(tokens):
            break
    return chunks


def process_pdf(name: str, data: bytes) -> list[dict]:
    """Chunks for one PDF: [{'source', 'page', 'text'}, ...]. Empty list = no readable text."""
    chunks = []
    for page, text in extract_pages(data):
        for piece in chunk_text(text):
            chunks.append({"source": name, "page": page, "text": piece})
    return chunks
