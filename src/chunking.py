"""
Text Chunking Module for HR Policy Documents.

Splits document text into manageable chunks using RecursiveCharacterTextSplitter
while preserving critical citation metadata: source filename, document title,
original PDF page number, and chunk index.
"""

import logging
from typing import Any, Dict, List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120


def get_text_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> RecursiveCharacterTextSplitter:
    """
    Creates and returns a configured RecursiveCharacterTextSplitter instance.
    Uses standard separators ordered from paragraphs down to words/characters
    to preserve cohesive semantic units.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        is_separator_regex=False
    )


def chunk_page_text(
    page_text: str,
    page_number: int,
    filename: str,
    doc_title: str,
    splitter: Optional[RecursiveCharacterTextSplitter] = None
) -> List[Dict[str, Any]]:
    """
    Splits the text of a single PDF page into chunks, attaching page-level metadata.

    Returns a list of chunk dicts:
    [
        {
            "chunk_id": str (e.g. "remote_work_policy.pdf_p2_c0"),
            "text": str,
            "source_file": str,
            "doc_title": str,
            "page_number": int,
            "char_count": int
        },
        ...
    ]
    """
    if not page_text or not page_text.strip():
        return []

    if splitter is None:
        splitter = get_text_splitter()

    raw_chunks = splitter.split_text(page_text)
    chunks: List[Dict[str, Any]] = []

    for idx, chunk_str in enumerate(raw_chunks):
        cleaned_chunk = chunk_str.strip()
        if not cleaned_chunk:
            continue

        chunk_id = f"{filename}_p{page_number}_c{idx}"
        chunks.append({
            "chunk_id": chunk_id,
            "text": cleaned_chunk,
            "source_file": filename,
            "doc_title": doc_title,
            "page_number": page_number,
            "char_count": len(cleaned_chunk)
        })

    return chunks


def chunk_document(
    document: Dict[str, Any],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[Dict[str, Any]]:
    """
    Chunks all pages of a structured document dictionary produced by ingest.py.
    """
    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    filename = document["filename"]
    doc_title = document.get("title", filename)
    all_chunks: List[Dict[str, Any]] = []

    for page in document.get("pages", []):
        page_num = page["page_number"]
        page_text = page["text"]
        page_chunks = chunk_page_text(
            page_text=page_text,
            page_number=page_num,
            filename=filename,
            doc_title=doc_title,
            splitter=splitter
        )
        all_chunks.extend(page_chunks)

    logger.info(
        f"Document '{filename}' ({len(document.get('pages', []))} pages) "
        f"split into {len(all_chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})."
    )
    return all_chunks


def chunk_all_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[Dict[str, Any]]:
    """
    Iterates over all ingested documents and chunks their pages.
    """
    all_chunks: List[Dict[str, Any]] = []
    for doc in documents:
        if doc.get("is_scanned_or_empty"):
            logger.warning(f"Skipping empty or scanned document '{doc['filename']}'")
            continue
        chunks = chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks.extend(chunks)

    return all_chunks
