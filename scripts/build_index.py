"""
Index Builder Script for HR Policy RAG Assistant.

Ingests policy PDFs from data/policies/, creates chunks, generates embeddings,
and saves the FAISS index to data/faiss_index/.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingest import download_default_policies, load_all_policies
from src.chunking import chunk_all_documents
from src.vector_store import FAISSVectorStore


def main():
    print("--- 1. Checking / Downloading Policy Documents ---")
    download_default_policies()

    print("\n--- 2. Ingesting Policy PDFs ---")
    documents = load_all_policies()
    print(f"Loaded {len(documents)} documents:")
    for doc in documents:
        total_chars = sum(p["char_count"] for p in doc["pages"])
        print(f" - {doc['filename']}: {len(doc['pages'])} pages, {total_chars} total characters")

    print("\n--- 3. Chunking Documents ---")
    chunks = chunk_all_documents(documents)
    print(f"Generated {len(chunks)} total chunks.")

    print("\n--- 4. Building FAISS Vector Index ---")
    store = FAISSVectorStore(index_dir="data/faiss_index")
    store.build_index(chunks)
    print(f"Successfully built and persisted FAISS index with {store.count()} vectors.")


if __name__ == "__main__":
    main()
