"""
Retrieval Module for HR Policy RAG Assistant.

Queries the FAISS vector store, formats retrieved context passages for the LLM prompt,
and prepares verifiable citation objects for user inspection.
"""

import logging
from typing import Any, Dict, List, Optional
from src.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 4


def retrieve_context(
    query: str,
    vector_store: FAISSVectorStore,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Performs similarity search on the vector store for the query and
    returns enriched retrieval results with citation tags.

    Each returned dictionary contains:
    {
        "chunk_id": str,
        "text": str,
        "source_file": str,
        "doc_title": str,
        "page_number": int,
        "similarity_score": float,
        "citation_tag": str  (e.g., "[Dickinson College Vacation Policy, p. 2]")
    }
    """
    cleaned_query = query.strip() if query else ""
    if not cleaned_query:
        logger.warning("Empty query passed to retrieve_context.")
        return []

    search_hits = vector_store.similarity_search(
        query=cleaned_query,
        top_k=top_k,
        score_threshold=score_threshold
    )

    enriched_results: List[Dict[str, Any]] = []
    for hit_chunk, score in search_hits:
        title = hit_chunk.get("doc_title", hit_chunk.get("source_file", "Unknown Document"))
        page = hit_chunk.get("page_number", 1)
        tag = f"[{title}, p. {page}]"

        enriched_results.append({
            "chunk_id": hit_chunk.get("chunk_id", ""),
            "text": hit_chunk.get("text", ""),
            "source_file": hit_chunk.get("source_file", ""),
            "doc_title": title,
            "page_number": page,
            "similarity_score": round(score, 4),
            "citation_tag": tag
        })

    logger.info(f"Retrieved {len(enriched_results)} chunks for query: '{cleaned_query[:40]}...'")
    return enriched_results


def format_context_for_prompt(retrieved_items: List[Dict[str, Any]]) -> str:
    """
    Formats retrieved chunks into a structured, numbered context string
    suitable for feeding into the LLM system/user prompt.
    """
    if not retrieved_items:
        return "No relevant policy documents or excerpts were found for this query."

    context_blocks = []
    for i, item in enumerate(retrieved_items, start=1):
        block = (
            f"--- SOURCE {i} ---\n"
            f"Document Title: {item['doc_title']}\n"
            f"Filename: {item['source_file']}\n"
            f"Page Number: {item['page_number']}\n"
            f"Passage Content:\n{item['text']}\n"
        )
        context_blocks.append(block)

    return "\n".join(context_blocks)


def extract_verifiable_citations(retrieved_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Builds clean, structured citation dictionaries from the retrieved chunks.
    Ensures the user interface only cites documents and pages that actually
    exist in the retrieved context.
    """
    citations = []
    seen = set()

    for item in retrieved_items:
        key = (item["source_file"], item["page_number"])
        if key in seen:
            continue
        seen.add(key)

        # Generate a concise snippet (first 180 characters)
        raw_text = item["text"]
        excerpt = (raw_text[:180] + "...") if len(raw_text) > 180 else raw_text

        citations.append({
            "doc_title": item["doc_title"],
            "source_file": item["source_file"],
            "page_number": item["page_number"],
            "similarity_score": item["similarity_score"],
            "excerpt": excerpt,
            "full_text": raw_text
        })

    return citations
