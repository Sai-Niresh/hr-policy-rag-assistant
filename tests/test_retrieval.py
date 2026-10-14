"""
Unit tests for vector indexing, persistence, and retrieval formatting.
"""

import os
import shutil
import tempfile
import pytest
from src.vector_store import FAISSVectorStore
from src.retrieval import (
    extract_verifiable_citations,
    format_context_for_prompt,
    retrieve_context,
)


@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id": "remote_p1_c0",
            "text": "Employees who work remotely must be reachable during core business hours from 8:30 AM to 4:30 PM.",
            "source_file": "remote_work_policy.pdf",
            "doc_title": "Remote Work Policy",
            "page_number": 1,
            "char_count": 97
        },
        {
            "chunk_id": "sick_p1_c0",
            "text": "The sick leave bank provides up to 6 months of paid leave for qualifying medical emergencies.",
            "source_file": "sick_leave_bank.pdf",
            "doc_title": "Sick Leave Bank Policy",
            "page_number": 1,
            "char_count": 93
        },
        {
            "chunk_id": "vacation_p1_c0",
            "text": "Full-time nonexempt employees earn 15 days of vacation per year for the first 10 years of service.",
            "source_file": "vacation.pdf",
            "doc_title": "Vacation Policy",
            "page_number": 1,
            "char_count": 98
        }
    ]


@pytest.fixture
def temp_vector_store(sample_chunks):
    temp_dir = tempfile.mkdtemp()
    store = FAISSVectorStore(index_dir=temp_dir)
    store.build_index(sample_chunks)
    yield store
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_vector_store_build_and_search(temp_vector_store):
    """Test index creation and top-k semantic search."""
    assert temp_vector_store.count() == 3
    results = temp_vector_store.similarity_search("How many months of sick leave can I get?", top_k=1)
    assert len(results) == 1
    hit_chunk, score = results[0]
    assert hit_chunk["source_file"] == "sick_leave_bank.pdf"
    assert "sick leave bank" in hit_chunk["text"].lower()
    assert score > 0.3


def test_vector_store_persistence(temp_vector_store, sample_chunks):
    """Test saving to disk and re-loading into a fresh instance."""
    index_dir = temp_vector_store.index_dir

    new_store = FAISSVectorStore(index_dir=index_dir)
    loaded = new_store.load_local()
    assert loaded is True
    assert new_store.count() == 3
    assert set(new_store.get_indexed_files()) == {
        "remote_work_policy.pdf",
        "sick_leave_bank.pdf",
        "vacation.pdf"
    }


def test_retrieve_context_empty_query(temp_vector_store):
    """Empty query should safely return an empty list."""
    assert retrieve_context("", temp_vector_store) == []
    assert retrieve_context("   ", temp_vector_store) == []


def test_format_context_for_prompt():
    """Verify structured prompt block formatting."""
    items = [
        {
            "chunk_id": "c1",
            "doc_title": "Vacation Policy",
            "source_file": "vacation.pdf",
            "page_number": 2,
            "text": "Employees cannot roll over unused vacation days."
        }
    ]
    prompt_str = format_context_for_prompt(items)
    assert "--- SOURCE 1 ---" in prompt_str
    assert "Document Title: Vacation Policy" in prompt_str
    assert "Page Number: 2" in prompt_str
    assert "Employees cannot roll over" in prompt_str


def test_extract_verifiable_citations():
    """Verify clean citation generation and deduplication by (source_file, page)."""
    items = [
        {
            "doc_title": "Vacation Policy",
            "source_file": "vacation.pdf",
            "page_number": 1,
            "similarity_score": 0.85,
            "text": "Nonexempt employees receive 15 days."
        },
        {
            "doc_title": "Vacation Policy",
            "source_file": "vacation.pdf",
            "page_number": 1,
            "similarity_score": 0.75,
            "text": "Second chunk on page 1."
        }
    ]
    citations = extract_verifiable_citations(items)
    assert len(citations) == 1
    assert citations[0]["doc_title"] == "Vacation Policy"
    assert citations[0]["page_number"] == 1
    assert citations[0]["excerpt"] == "Nonexempt employees receive 15 days."
