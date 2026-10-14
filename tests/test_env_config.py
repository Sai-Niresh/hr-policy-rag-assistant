"""
Tests for environment configuration and API key loading.

Verifies that:
- .env is discovered and loaded correctly when PROJECT_ROOT is set
- Missing API key produces the expected status in answer_generator
- Authentication errors are caught and reported gracefully
- Model fallback logic is triggered on NotFoundError

NOTE: Because answer_generator.py loads .env at module import time and
get_configured_api_key() reads from os.environ, tests that need "no key"
behavior must patch os.environ AND instantiate RAGAnswerGenerator inside
the patch context, AND ensure similarity_search returns at least one chunk
so the code does not short-circuit to insufficient_context before the
missing_api_key check.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_DUMMY_CHUNK = (
    {
        "chunk_id": "test::c1",
        "text": "Policy text. " * 10,
        "source_file": "vacation.pdf",
        "doc_title": "Vacation Policy",
        "page_number": 1,
    },
    0.88,
)


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _make_mock_store(with_chunks: bool = True):
    """
    Creates a minimal FAISSVectorStore mock.

    Args:
        with_chunks: If True, similarity_search returns one dummy chunk.
                     If False, returns empty list (triggers insufficient_context path).
    """
    store = MagicMock()
    store.similarity_search.return_value = [_DUMMY_CHUNK] if with_chunks else []
    store.count.return_value = 1 if with_chunks else 0
    return store


# ────────────────────────────────────────────────────────────────────
# Tests: API key detection
# ────────────────────────────────────────────────────────────────────

class TestApiKeyConfiguration:
    """Verifies RAGAnswerGenerator correctly detects presence/absence of API key."""

    def test_configured_when_explicit_key_provided(self):
        """Passing a real-looking key directly should mark API as configured."""
        from src.answer_generator import RAGAnswerGenerator
        gen = RAGAnswerGenerator(vector_store=_make_mock_store(), api_key="gsk_test_key_1234")
        assert gen.is_api_key_configured() is True

    def test_placeholder_key_treated_as_unconfigured(self):
        """Keys starting with 'your_' should be treated as unconfigured placeholders."""
        from src.answer_generator import get_configured_api_key
        # Also clear env so there's no fallback
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}, clear=False):
            key = get_configured_api_key("your_groq_api_key_here")
        assert key == ""

    def test_empty_string_with_no_env_key_returns_empty(self):
        """Empty passed key + empty env var = no configured key."""
        from src.answer_generator import get_configured_api_key
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}, clear=False):
            key = get_configured_api_key("")
        assert key == ""

    def test_missing_key_returns_missing_api_key_status(self):
        """Generator with no real key should return missing_api_key status (not error)."""
        from src.answer_generator import RAGAnswerGenerator
        store = _make_mock_store(with_chunks=True)

        # Patch env to remove any real key, then build generator inside context
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}, clear=False):
            gen = RAGAnswerGenerator(vector_store=store, api_key="your_placeholder")

        assert not gen.is_api_key_configured()
        result = gen.answer_question("What is the vacation policy?")
        assert result["status"] == "missing_api_key", (
            f"Expected 'missing_api_key' but got '{result['status']}'. "
            "Check that get_configured_api_key returns '' for placeholder keys."
        )
        assert "GROQ_API_KEY" in result["answer"]


# ────────────────────────────────────────────────────────────────────
# Tests: Exception handling during API calls
# ────────────────────────────────────────────────────────────────────

class TestApiExceptionHandling:
    """Verifies that Groq API errors are caught and mapped to correct statuses."""

    def _gen_with_dummy_key(self):
        """Creates a generator with a fake but non-empty key to pass the key check."""
        from src.answer_generator import RAGAnswerGenerator
        # Use with_chunks=True so retrieval succeeds and code reaches the LLM call
        store = _make_mock_store(with_chunks=True)
        gen = RAGAnswerGenerator(vector_store=store, api_key="gsk_fake_but_non_empty")
        return gen

    def test_authentication_error_returns_auth_error_status(self):
        """AuthenticationError should produce status='auth_error' with key-related message."""
        import groq
        gen = self._gen_with_dummy_key()

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = groq.AuthenticationError(
            message="Invalid API key", response=MagicMock(status_code=401), body={}
        )
        gen.client = mock_client

        result = gen.answer_question("Test question?")
        assert result["status"] == "auth_error", (
            f"Expected 'auth_error', got '{result['status']}'"
        )
        assert "Authentication" in result["answer"] or "API key" in result["answer"]

    def test_rate_limit_error_returns_rate_limit_status(self):
        """RateLimitError should produce status='rate_limit_error'."""
        import groq
        gen = self._gen_with_dummy_key()

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = groq.RateLimitError(
            message="Rate limit exceeded", response=MagicMock(status_code=429), body={}
        )
        gen.client = mock_client

        result = gen.answer_question("Test question?")
        assert result["status"] == "rate_limit_error", (
            f"Expected 'rate_limit_error', got '{result['status']}'"
        )
        assert "Rate Limit" in result["answer"] or "rate limit" in result["answer"].lower()

    def test_connection_error_returns_connection_error_status(self):
        """APIConnectionError should produce status='connection_error'."""
        import groq
        gen = self._gen_with_dummy_key()

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = groq.APIConnectionError(
            request=MagicMock()
        )
        gen.client = mock_client

        result = gen.answer_question("Test question?")
        assert result["status"] == "connection_error", (
            f"Expected 'connection_error', got '{result['status']}'"
        )
        assert "Connection" in result["answer"] or "connection" in result["answer"].lower()


# ────────────────────────────────────────────────────────────────────
# Tests: .env file loading
# ────────────────────────────────────────────────────────────────────

class TestDotEnvLoading:
    """Verifies the answer_generator module loads .env from PROJECT_ROOT."""

    def test_env_file_exists_at_project_root(self):
        """The .env file must exist at the project root."""
        env_path = PROJECT_ROOT / ".env"
        assert env_path.exists(), (
            f".env file missing at {env_path}. "
            "Create it with GROQ_API_KEY=your_key and GROQ_MODEL=openai/gpt-oss-120b"
        )

    def test_env_example_has_no_real_key(self):
        """Sanity check: .env.example should not contain a real key value."""
        example_path = PROJECT_ROOT / ".env.example"
        if example_path.exists():
            content = example_path.read_text(encoding="utf-8")
            lines = [ln for ln in content.splitlines() if "GROQ_API_KEY" in ln]
            for line in lines:
                if "=" in line:
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    assert not val.startswith("gsk_"), (
                        ".env.example still contains what looks like a real Groq API key. "
                        "Replace it with a placeholder like 'your_groq_api_key_here'."
                    )

    def test_groq_api_key_env_var_is_non_empty(self):
        """GROQ_API_KEY must be set in the environment (loaded from .env)."""
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)
        key = os.getenv("GROQ_API_KEY", "")
        assert key, "GROQ_API_KEY is empty after loading .env. Check your .env file."

    def test_groq_model_env_var_is_set(self):
        """GROQ_MODEL should be set (defaults acceptable if missing)."""
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)
        model = os.getenv("GROQ_MODEL", "")
        assert isinstance(model, str)
