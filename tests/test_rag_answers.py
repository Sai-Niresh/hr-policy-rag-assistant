"""
Unit and integration tests for RAG answer generation, prompt construction, and citations.
"""

from unittest.mock import MagicMock, patch
import pytest
from src.answer_generator import (
    ABSTENTION_MESSAGE,
    RAGAnswerGenerator,
    build_user_prompt,
)
from src.vector_store import FAISSVectorStore


@pytest.fixture
def mock_vector_store():
    store = MagicMock(spec=FAISSVectorStore)
    store.similarity_search.return_value = [
        (
            {
                "chunk_id": "test_c1",
                "text": "The maximum payout from the sick leave bank is capped at six months.",
                "source_file": "sick_leave_bank.pdf",
                "doc_title": "Dickinson College Sick-Leave Bank Policy",
                "page_number": 1,
            },
            0.88
        )
    ]
    return store


def test_build_user_prompt():
    """Verify prompt combines question and formatted context."""
    prompt = build_user_prompt("What is the limit?", "--- SOURCE 1 ---\nLimit is 6 months.")
    assert "User Question: What is the limit?" in prompt
    assert "Retrieved Policy Excerpts:" in prompt
    assert "Limit is 6 months." in prompt


def test_answer_empty_query(mock_vector_store):
    """Test behavior on empty user query."""
    generator = RAGAnswerGenerator(vector_store=mock_vector_store, api_key="dummy_key")
    res = generator.answer_question("")
    assert res["status"] == "empty_query"
    assert "valid question" in res["answer"].lower()


def test_answer_missing_api_key(mock_vector_store):
    """Test graceful handling and instruction message when API key is missing."""
    # Patch env so get_configured_api_key cannot fall back to a real key
    with patch.dict("os.environ", {"GROQ_API_KEY": ""}, clear=False):
        generator = RAGAnswerGenerator(vector_store=mock_vector_store, api_key="")
    # generator.api_key should be "" since both passed key and env are empty
    assert not generator.is_api_key_configured()
    res = generator.answer_question("What is the policy limit?")
    assert res["status"] == "missing_api_key", (
        f"Expected 'missing_api_key', got '{res['status']}'. "
        "Ensure GROQ_API_KEY is not leaking from environment into the test."
    )
    assert "GROQ_API_KEY is not configured" in res["answer"]
    assert len(res["retrieved_chunks"]) == 1
    assert len(res["citations"]) == 1


def test_answer_with_mocked_llm(mock_vector_store):
    """Test full RAG generation pipeline with mocked Groq completion."""
    generator = RAGAnswerGenerator(vector_store=mock_vector_store, api_key="gsk_test123")

    mock_choice = MagicMock()
    mock_choice.message.content = (
        "The maximum payout from the Sick Leave Bank is six months "
        "[Dickinson College Sick-Leave Bank Policy, p. 1]."
    )
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    generator.client = mock_client

    res = generator.answer_question("What is the maximum sick leave payout?")

    assert res["status"] == "success"
    assert "six months" in res["answer"]
    assert len(res["citations"]) == 1
    assert res["citations"][0]["doc_title"] == "Dickinson College Sick-Leave Bank Policy"
    assert res["citations"][0]["page_number"] == 1
    assert res["is_abstention"] is False


def test_abstention_detection(mock_vector_store):
    """Verify that abstention response correctly sets is_abstention=True."""
    generator = RAGAnswerGenerator(vector_store=mock_vector_store, api_key="gsk_test123")

    mock_choice = MagicMock()
    mock_choice.message.content = (
        "I am sorry, but the provided HR policy documents do not contain "
        "sufficient information to establish an answer to this question."
    )
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    generator.client = mock_client

    res = generator.answer_question("What is the pet insurance policy?")

    assert res["status"] == "success"
    assert res["is_abstention"] is True
    assert len(res["citations"]) == 0
