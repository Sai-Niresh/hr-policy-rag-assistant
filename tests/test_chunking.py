"""
Unit tests for text chunking and metadata preservation.
"""

import pytest
from src.chunking import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_all_documents,
    chunk_document,
    chunk_page_text,
    get_text_splitter,
)


def test_chunk_page_text_normal():
    """Verify that normal text is split with all metadata preserved."""
    sample_text = (
        "Dickinson College establishes this remote work policy to outline expectations "
        "for eligible employees. Remote work is a voluntary alternative work arrangement. "
        "Employees must maintain the same standard of performance as in-office employees. "
    ) * 10  # ~1600 characters

    chunks = chunk_page_text(
        page_text=sample_text,
        page_number=2,
        filename="remote_policy.pdf",
        doc_title="Remote Policy"
    )

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk["page_number"] == 2
        assert chunk["source_file"] == "remote_policy.pdf"
        assert chunk["doc_title"] == "Remote Policy"
        assert "remote_policy.pdf_p2_c" in chunk["chunk_id"]
        assert len(chunk["text"]) <= DEFAULT_CHUNK_SIZE + 100
        assert chunk["char_count"] == len(chunk["text"])


def test_chunk_page_text_empty_and_whitespace():
    """Verify that empty or whitespace-only pages return empty lists."""
    assert chunk_page_text("", 1, "test.pdf", "Test") == []
    assert chunk_page_text("   \n\n  \t  ", 1, "test.pdf", "Test") == []


def test_chunk_document_preserves_page_numbers():
    """Verify that chunks across multiple pages retain correct page numbers."""
    doc = {
        "filename": "vacation.pdf",
        "title": "Vacation Policy",
        "pages": [
            {"page_number": 1, "text": "Page 1 content on nonexempt vacation accruals."},
            {"page_number": 2, "text": "Page 2 content on exempt staff vacation limits."}
        ]
    }

    chunks = chunk_document(doc, chunk_size=400, chunk_overlap=50)
    assert len(chunks) == 2
    assert chunks[0]["page_number"] == 1
    assert chunks[0]["text"] == "Page 1 content on nonexempt vacation accruals."
    assert chunks[1]["page_number"] == 2
    assert chunks[1]["text"] == "Page 2 content on exempt staff vacation limits."


def test_chunk_all_documents_skips_scanned_empty():
    """Verify that empty/scanned documents are safely skipped without error."""
    docs = [
        {
            "filename": "empty.pdf",
            "title": "Empty PDF",
            "pages": [],
            "is_scanned_or_empty": True
        },
        {
            "filename": "valid.pdf",
            "title": "Valid PDF",
            "pages": [{"page_number": 1, "text": "Valid text policy content."}],
            "is_scanned_or_empty": False
        }
    ]

    all_chunks = chunk_all_documents(docs)
    assert len(all_chunks) == 1
    assert all_chunks[0]["source_file"] == "valid.pdf"
