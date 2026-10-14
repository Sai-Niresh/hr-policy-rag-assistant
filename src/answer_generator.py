"""
LLM Answer Generation Module for HR Policy RAG Assistant.

Interacts with the Groq API to generate factual, grounded answers with verifiable inline citations.
Enforces strict context adherence and explicit abstention when information is absent.
Provides resilient .env path resolution, Streamlit secrets fallback, and robust error classification.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import groq
from groq import Groq

from src.retrieval import (
    DEFAULT_TOP_K,
    extract_verifiable_citations,
    format_context_for_prompt,
    retrieve_context,
)
from src.vector_store import FAISSVectorStore

# Resolve project root explicitly regardless of execution CWD
PROJECT_ROOT = Path(__file__).resolve().parent.parent
for env_candidate in [PROJECT_ROOT / ".env", PROJECT_ROOT / ".env.txt"]:
    if env_candidate.exists():
        load_dotenv(dotenv_path=env_candidate, override=True)
        break

logger = logging.getLogger(__name__)

DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
ABSTENTION_MESSAGE = (
    "I am sorry, but the provided HR policy documents do not contain "
    "sufficient information to establish an answer to this question."
)

SYSTEM_PROMPT = """You are an official HR Policy Assistant for Dickinson College.
Your sole job is to answer employee questions strictly and accurately using the provided policy excerpts.

RULES:
1. ONLY use information directly stated in the provided context below. Do NOT use outside knowledge, assumptions, or extrapolations.
2. If the answer cannot be definitively established from the provided excerpts, state:
   "I am sorry, but the provided HR policy documents do not contain sufficient information to establish an answer to this question."
3. Every factual claim MUST be followed by an exact source citation in this format: [Document Title, p. X].
4. Do NOT invent, assume, or hallucinate page numbers, document titles, or policy provisions.
5. Keep your answer clear, professional, concise, and structured.
"""


def get_configured_api_key(passed_key: Optional[str] = None) -> str:
    """
    Safely retrieves the Groq API key from passed arguments, environment,
    or Streamlit secrets without exposing the secret.
    """
    if passed_key and passed_key.strip() and not passed_key.strip().startswith("your_"):
        return passed_key.strip()

    env_key = os.getenv("GROQ_API_KEY", "").strip()
    if env_key and not env_key.startswith("your_"):
        return env_key

    try:
        import streamlit as st
        secret_key = st.secrets.get("GROQ_API_KEY", "").strip()
        if secret_key and not secret_key.startswith("your_"):
            return secret_key
    except Exception:
        pass

    return ""


def build_user_prompt(question: str, context_str: str) -> str:
    """Combines user question and formatted retrieved context."""
    return f"""Retrieved Policy Excerpts:
{context_str}

User Question: {question}

Provide an accurate, policy-grounded answer based strictly on the excerpts above with citations in format [Document Title, p. X]:"""


class RAGAnswerGenerator:
    """
    Orchestrates context retrieval, prompt construction, and Groq LLM answer generation.
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.vector_store = vector_store
        self.api_key = get_configured_api_key(api_key)
        self.model_name = model_name or os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
        self._client: Optional[Groq] = None

    @property
    def client(self) -> Groq:
        """Initializes Groq client with API key."""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "GROQ_API_KEY is not set. Please supply your Groq API key in .env or the UI."
                )
            self._client = Groq(api_key=self.api_key)
        return self._client

    @client.setter
    def client(self, value: Any) -> None:
        """Allows setting or mocking the client."""
        self._client = value

    def is_api_key_configured(self) -> bool:
        """Checks if a valid non-placeholder API key is available."""
        return bool(self.api_key and not self.api_key.startswith("your_"))

    def answer_question(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """
        Executes end-to-end RAG answering pipeline for a user question.
        """
        cleaned_question = question.strip() if question else ""
        if not cleaned_question:
            return {
                "question": question,
                "answer": "Please enter a valid question regarding HR policies.",
                "citations": [],
                "retrieved_chunks": [],
                "is_abstention": False,
                "status": "empty_query"
            }

        # 1. Retrieve top-k context passages from local FAISS vector store
        retrieved_chunks = retrieve_context(
            query=cleaned_question,
            vector_store=self.vector_store,
            top_k=top_k
        )

        citations = extract_verifiable_citations(retrieved_chunks)

        # 2. Check if vector store has no entries or retrieved nothing
        if not retrieved_chunks:
            return {
                "question": cleaned_question,
                "answer": ABSTENTION_MESSAGE,
                "citations": [],
                "retrieved_chunks": [],
                "is_abstention": True,
                "status": "insufficient_context"
            }

        # 3. Check API key presence
        if not self.is_api_key_configured():
            logger.warning("GROQ_API_KEY is not configured.")
            return {
                "question": cleaned_question,
                "answer": (
                    "⚠️ **GROQ_API_KEY is not configured.**\n\n"
                    "The local RAG pipeline successfully extracted and retrieved the top policy passages below, "
                    "but generating the final synthesized answer requires an active Groq API key.\n\n"
                    "**How to configure:**\n"
                    "1. Obtain a free API key from [Groq Console](https://console.groq.com/keys).\n"
                    "2. Add `GROQ_API_KEY=your_key_here` to your `.env` file or enter it in the sidebar settings.\n"
                    "3. Restart the Streamlit application."
                ),
                "citations": citations,
                "retrieved_chunks": retrieved_chunks,
                "is_abstention": False,
                "status": "missing_api_key"
            }

        # 4. Format prompt context and query LLM
        context_str = format_context_for_prompt(retrieved_chunks)
        user_prompt = build_user_prompt(cleaned_question, context_str)

        try:
            # Attempt generation with configured model, with automatic model fallback
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    max_tokens=600
                )
            except groq.NotFoundError:
                # If requested model is not found, discover an accessible chat model
                logger.warning(f"Model '{self.model_name}' not accessible; querying available models...")
                available = self.client.models.list()
                chat_candidates = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"]
                fallback_model = None
                available_ids = [m.id for m in available.data]
                for candidate in chat_candidates:
                    if candidate in available_ids:
                        fallback_model = candidate
                        break
                if not fallback_model and available_ids:
                    fallback_model = available_ids[0]

                if not fallback_model:
                    raise

                logger.info(f"Fell back to model: {fallback_model}")
                self.model_name = fallback_model
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    max_tokens=600
                )

            raw_answer = response.choices[0].message.content.strip()

            # Determine if model indicated abstention
            is_abstention = (
                "do not contain sufficient information" in raw_answer.lower()
                or "cannot establish" in raw_answer.lower()
                or "not mentioned in the provided" in raw_answer.lower()
                or "not mentioned" in raw_answer.lower()
            )

            return {
                "question": cleaned_question,
                "answer": raw_answer,
                "citations": citations if not is_abstention else [],
                "retrieved_chunks": retrieved_chunks,
                "is_abstention": is_abstention,
                "status": "success"
            }

        except groq.AuthenticationError:
            err_msg = "Authentication Error: The provided GROQ_API_KEY is invalid or expired. Please verify your credentials."
            logger.error(err_msg)
            return {
                "question": cleaned_question,
                "answer": f"⚠️ **{err_msg}**",
                "citations": citations,
                "retrieved_chunks": retrieved_chunks,
                "is_abstention": False,
                "status": "auth_error"
            }
        except groq.RateLimitError:
            err_msg = "Rate Limit Error: Groq API request quota exceeded. Please wait a moment before trying again."
            logger.error(err_msg)
            return {
                "question": cleaned_question,
                "answer": f"⚠️ **{err_msg}**",
                "citations": citations,
                "retrieved_chunks": retrieved_chunks,
                "is_abstention": False,
                "status": "rate_limit_error"
            }
        except groq.APIConnectionError:
            err_msg = "Connection Error: Failed to connect to Groq API servers. Please check your network connection."
            logger.error(err_msg)
            return {
                "question": cleaned_question,
                "answer": f"⚠️ **{err_msg}**",
                "citations": citations,
                "retrieved_chunks": retrieved_chunks,
                "is_abstention": False,
                "status": "connection_error"
            }
        except Exception as e:
            err_msg = f"API Error: Failed to generate response ({str(e)})"
            logger.error(err_msg)
            return {
                "question": cleaned_question,
                "answer": f"⚠️ **{err_msg}**",
                "citations": citations,
                "retrieved_chunks": retrieved_chunks,
                "is_abstention": False,
                "status": "error"
            }
