"""
Document loaders for knowledge base ingestion.

Supports:
  .pdf   — FSA Handbook chapters, federal regulations
  .docx  — university policy documents
  .txt   — plain-text documents
  .json  — FAQ databases (list of {question, answer, source} dicts)
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Iterator


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split text into overlapping chunks respecting sentence boundaries where possible."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break
        # Prefer breaking at a sentence boundary
        boundary = text.rfind(". ", start, end)
        if boundary == -1 or boundary < start + chunk_size // 2:
            boundary = end
        else:
            boundary += 2  # include the period and space
        chunks.append(text[start:boundary])
        start = boundary - overlap
    return [c.strip() for c in chunks if c.strip()]


def _doc_id(content: str, source: str) -> str:
    digest = hashlib.sha256(f"{source}::{content[:200]}".encode()).digest()
    return str(uuid.UUID(bytes=digest[:16]))


def load_pdf(path: Path, doc_type: str = "fsa_handbook") -> Iterator[dict]:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    for page_num, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        source = f"{path.name} — page {page_num}"
        for chunk in _chunk_text(text):
            yield {
                "id": _doc_id(chunk, source),
                "content": chunk,
                "source": source,
                "doc_type": doc_type,
                "metadata": {"file": path.name, "page": page_num},
            }


def load_docx(path: Path, doc_type: str = "university_policy") -> Iterator[dict]:
    from docx import Document
    doc = Document(str(path))
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    source = path.name
    for chunk in _chunk_text(full_text):
        yield {
            "id": _doc_id(chunk, source),
            "content": chunk,
            "source": source,
            "doc_type": doc_type,
            "metadata": {"file": path.name},
        }


def load_txt(path: Path, doc_type: str = "fsa_handbook") -> Iterator[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    source = path.name
    for chunk in _chunk_text(text):
        yield {
            "id": _doc_id(chunk, source),
            "content": chunk,
            "source": source,
            "doc_type": doc_type,
            "metadata": {"file": path.name},
        }


def load_faq_json(path: Path) -> Iterator[dict]:
    """
    Load structured FAQ JSON.

    Expected format:
      [{"question": "...", "answer": "...", "source": "...", "category": "..."}]
    """
    faqs = json.loads(path.read_text(encoding="utf-8"))
    for faq in faqs:
        content = f"Q: {faq['question']}\nA: {faq['answer']}"
        source = faq.get("source", path.name)
        yield {
            "id": _doc_id(content, source),
            "content": content,
            "source": source,
            "doc_type": "faq",
            "metadata": {
                "question": faq["question"],
                "category": faq.get("category", "general"),
            },
        }


def load_directory(directory: Path) -> list[dict]:
    """
    Recursively load all supported documents from a directory tree.
    Infers doc_type from subdirectory name.
    """
    docs = []
    for path in directory.rglob("*"):
        if not path.is_file():
            continue

        # Infer doc_type from parent folder name
        parent = path.parent.name
        if parent == "faqs":
            doc_type = "faq"
        elif parent == "university_policies":
            doc_type = "university_policy"
        else:
            doc_type = "fsa_handbook"

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            docs.extend(load_pdf(path, doc_type))
        elif suffix == ".docx":
            docs.extend(load_docx(path, doc_type))
        elif suffix == ".txt":
            docs.extend(load_txt(path, doc_type))
        elif suffix == ".json" and parent == "faqs":
            docs.extend(load_faq_json(path))

    return docs
