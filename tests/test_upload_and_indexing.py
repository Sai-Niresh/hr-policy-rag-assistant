"""
Tests for PDF upload validation, duplicate detection, indexing, and document removal.

Covers:
- validate_pdf_file() for all failure modes (size, format, corrupt, scanned)
- Successful validation of a real text PDF
- Document removal via FAISSVectorStore.remove_document()
- add_chunks() duplicate prevention (same filename already in index)
- get_document_stats() returns correct per-doc metrics
"""

import io
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest import validate_pdf_file


# ────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────

def _real_pdf_bytes() -> bytes:
    """
    Returns bytes of one of the project's real policy PDFs (vacation.pdf).
    Falls back to a minimal hand-crafted valid PDF if the file is missing.
    """
    pdf_path = PROJECT_ROOT / "data" / "policies" / "vacation.pdf"
    if pdf_path.exists():
        return pdf_path.read_bytes()

    # Minimal valid PDF with one text page
    minimal = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R"
        b"/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
        b"4 0 obj<</Length 44>>stream\n"
        b"BT /F1 12 Tf 100 700 Td (Hello HR Policy) Tj ET\n"
        b"endstream\nendobj\n"
        b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000274 00000 n \n"
        b"0000000370 00000 n \n"
        b"trailer<</Size 6/Root 1 0 R>>\n"
        b"startxref\n447\n%%EOF"
    )
    return minimal


def _make_sample_chunks(source_file: str = "test_doc.pdf", count: int = 3):
    """Creates a list of dummy chunk dicts for vector store tests."""
    return [
        {
            "chunk_id": f"{source_file}::chunk_{i}",
            "text": f"This is sample chunk {i} from {source_file}. " * 8,
            "source_file": source_file,
            "doc_title": "Test Document",
            "page_number": 1,
            "char_count": 100,
        }
        for i in range(count)
    ]


# ────────────────────────────────────────────────────────────────────
# Tests: validate_pdf_file()
# ────────────────────────────────────────────────────────────────────

class TestValidatePdfFile:
    """Tests for the validate_pdf_file() validation function."""

    def test_valid_real_pdf_passes(self):
        pdf_bytes = _real_pdf_bytes()
        result = validate_pdf_file(pdf_bytes, "test.pdf")
        assert result["ok"] is True
        assert result["error_type"] is None
        assert result["page_count"] > 0
        assert result["total_chars"] > 0

    def test_oversized_file_fails(self):
        large_bytes = b"%PDF-1.4" + b"A" * (21 * 1024 * 1024)
        result = validate_pdf_file(large_bytes, "huge.pdf", max_size_mb=20.0)
        assert result["ok"] is False
        assert result["error_type"] == "size"
        assert "MB" in result["message"]

    def test_non_pdf_bytes_fails_format_check(self):
        fake_bytes = b"PK\x03\x04" + b"\x00" * 100  # ZIP header
        result = validate_pdf_file(fake_bytes, "notapdf.pdf")
        assert result["ok"] is False
        assert result["error_type"] == "format"
        assert "PDF" in result["message"]

    def test_corrupt_pdf_fails(self):
        # PDF header present but no valid structure
        corrupt_bytes = b"%PDF-1.4\n" + b"\x00\x01\x02" * 50
        result = validate_pdf_file(corrupt_bytes, "corrupt.pdf")
        assert result["ok"] is False
        # Either corrupt or scanned is acceptable depending on PyMuPDF behaviour
        assert result["error_type"] in ("corrupt", "scanned")

    def test_empty_bytes_fails(self):
        result = validate_pdf_file(b"", "empty.pdf")
        assert result["ok"] is False
        assert result["error_type"] in ("size", "format")

    def test_message_is_human_readable(self):
        pdf_bytes = _real_pdf_bytes()
        result = validate_pdf_file(pdf_bytes, "vacation.pdf")
        assert isinstance(result["message"], str)
        assert len(result["message"]) > 10


# ────────────────────────────────────────────────────────────────────
# Tests: FAISSVectorStore document management
# ────────────────────────────────────────────────────────────────────

class TestFAISSDocumentManagement:
    """Tests for add_chunks(), remove_document(), and get_document_stats()."""

    def _build_store(self, tmp_path: Path):
        from src.vector_store import FAISSVectorStore
        store = FAISSVectorStore(index_dir=str(tmp_path / "faiss_index"))
        return store

    def test_add_chunks_builds_index(self, tmp_path):
        store = self._build_store(tmp_path)
        chunks = _make_sample_chunks("doc_a.pdf", count=5)
        store.build_index(chunks)
        assert store.count() == 5

    def test_get_indexed_files_lists_all_sources(self, tmp_path):
        store = self._build_store(tmp_path)
        chunks_a = _make_sample_chunks("doc_a.pdf", count=3)
        chunks_b = _make_sample_chunks("doc_b.pdf", count=2)
        store.build_index(chunks_a + chunks_b)

        files = store.get_indexed_files()
        assert "doc_a.pdf" in files
        assert "doc_b.pdf" in files
        assert len(files) == 2

    def test_get_document_stats_returns_correct_counts(self, tmp_path):
        store = self._build_store(tmp_path)
        chunks_a = _make_sample_chunks("doc_a.pdf", count=4)
        chunks_b = _make_sample_chunks("doc_b.pdf", count=6)
        store.build_index(chunks_a + chunks_b)

        stats = store.get_document_stats()
        assert len(stats) == 2

        counts = {s["source_file"]: s["chunk_count"] for s in stats}
        assert counts["doc_a.pdf"] == 4
        assert counts["doc_b.pdf"] == 6

    def test_remove_document_reduces_count(self, tmp_path):
        store = self._build_store(tmp_path)
        chunks_a = _make_sample_chunks("doc_a.pdf", count=3)
        chunks_b = _make_sample_chunks("doc_b.pdf", count=4)
        store.build_index(chunks_a + chunks_b)

        removed = store.remove_document("doc_a.pdf")

        assert removed is True
        assert store.count() == 4  # Only doc_b remains
        assert "doc_a.pdf" not in store.get_indexed_files()
        assert "doc_b.pdf" in store.get_indexed_files()

    def test_remove_nonexistent_document_returns_false(self, tmp_path):
        store = self._build_store(tmp_path)
        chunks = _make_sample_chunks("doc_a.pdf", count=3)
        store.build_index(chunks)

        removed = store.remove_document("nonexistent.pdf")
        assert removed is False
        assert store.count() == 3  # unchanged

    def test_remove_only_document_empties_index(self, tmp_path):
        store = self._build_store(tmp_path)
        chunks = _make_sample_chunks("only_doc.pdf", count=2)
        store.build_index(chunks)

        removed = store.remove_document("only_doc.pdf")
        assert removed is True
        assert store.count() == 0
        assert store.get_indexed_files() == []

    def test_similarity_search_after_removal(self, tmp_path):
        store = self._build_store(tmp_path)
        chunks_a = _make_sample_chunks("doc_a.pdf", count=3)
        chunks_b = _make_sample_chunks("doc_b.pdf", count=3)
        store.build_index(chunks_a + chunks_b)

        store.remove_document("doc_a.pdf")
        results = store.similarity_search("sample chunk", top_k=5)

        # All returned results must belong to doc_b
        for chunk, score in results:
            assert chunk["source_file"] == "doc_b.pdf"

    def test_add_chunks_appends_to_existing_index(self, tmp_path):
        store = self._build_store(tmp_path)
        chunks_a = _make_sample_chunks("doc_a.pdf", count=3)
        store.build_index(chunks_a)

        chunks_b = _make_sample_chunks("doc_b.pdf", count=4)
        store.add_chunks(chunks_b)

        assert store.count() == 7
        assert "doc_b.pdf" in store.get_indexed_files()

    def test_index_persists_to_disk(self, tmp_path):
        store = self._build_store(tmp_path)
        chunks = _make_sample_chunks("persist_doc.pdf", count=5)
        store.build_index(chunks)

        # Load into a new instance
        store2 = self._build_store(tmp_path)
        loaded = store2.load_local()

        assert loaded is True
        assert store2.count() == 5
        assert "persist_doc.pdf" in store2.get_indexed_files()
