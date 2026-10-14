"""
Vector Store & Embeddings Module using Sentence-Transformers and FAISS CPU.

Computes dense vector representations of policy chunks and performs fast,
verifiable local cosine similarity search.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_INDEX_DIR = "data/faiss_index"


class FAISSVectorStore:
    """
    Manages local FAISS index with inner-product (cosine similarity) search
    and associated chunk metadata persistence.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        index_dir: str = DEFAULT_INDEX_DIR
    ):
        self.model_name = model_name
        self.index_dir = index_dir
        self.index: Optional[faiss.Index] = None
        self.chunks_metadata: List[Dict[str, Any]] = []
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy loader for SentenceTransformer to save memory until needed."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generates normalized embedding vectors for a list of strings.
        Normalizing embeddings ensures that faiss.IndexFlatIP computes cosine similarity.
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)

    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Builds a fresh FAISS index from a list of chunk dictionaries
        and saves it to disk.
        """
        if not chunks:
            logger.warning("No chunks provided to build_index.")
            return

        texts = [c["text"] for c in chunks]
        logger.info(f"Embedding {len(texts)} chunks using {self.model_name}...")
        embeddings = self.embed_texts(texts)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        self.chunks_metadata = [dict(c) for c in chunks]

        self.save_local()
        logger.info(f"Successfully built and persisted index with {self.index.ntotal} vectors.")

    def add_chunks(self, new_chunks: List[Dict[str, Any]]) -> None:
        """
        Appends new chunks to an existing index or builds one if absent.
        """
        if not new_chunks:
            return

        if self.index is None or len(self.chunks_metadata) == 0:
            self.build_index(new_chunks)
            return

        texts = [c["text"] for c in new_chunks]
        embeddings = self.embed_texts(texts)

        self.index.add(embeddings)
        self.chunks_metadata.extend([dict(c) for c in new_chunks])
        self.save_local()
        logger.info(f"Added {len(new_chunks)} chunks; index total is now {self.index.ntotal}.")

    def similarity_search(
        self,
        query: str,
        top_k: int = 4,
        score_threshold: Optional[float] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Searches the FAISS index for the top_k most similar chunks to query.

        Returns a list of tuples: (chunk_dict, cosine_similarity_score).
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Similarity search called on empty or uninitialized index.")
            return []

        query_vec = self.embed_texts([query])
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, k)

        results: List[Tuple[Dict[str, Any], float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            sim_score = float(score)
            if score_threshold is not None and sim_score < score_threshold:
                continue

            chunk_meta = dict(self.chunks_metadata[idx])
            results.append((chunk_meta, sim_score))

        return results

    def save_local(self, target_dir: Optional[str] = None) -> None:
        """
        Persists the FAISS index binary and metadata JSON to disk.
        """
        out_dir = target_dir or self.index_dir
        os.makedirs(out_dir, exist_ok=True)

        index_file = os.path.join(out_dir, "index.faiss")
        meta_file = os.path.join(out_dir, "chunks_metadata.json")

        if self.index is not None:
            faiss.write_index(self.index, index_file)

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump({
                "model_name": self.model_name,
                "total_chunks": len(self.chunks_metadata),
                "chunks": self.chunks_metadata
            }, f, indent=2)

    def load_local(self, target_dir: Optional[str] = None) -> bool:
        """
        Loads an existing FAISS index and metadata from disk if available.
        Returns True if successful, False otherwise.
        """
        load_dir = target_dir or self.index_dir
        index_file = os.path.join(load_dir, "index.faiss")
        meta_file = os.path.join(load_dir, "chunks_metadata.json")

        if not os.path.exists(index_file) or not os.path.exists(meta_file):
            return False

        try:
            self.index = faiss.read_index(index_file)
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.chunks_metadata = data.get("chunks", [])
            logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors from {load_dir}.")
            return True
        except Exception as e:
            logger.error(f"Failed to load FAISS index from {load_dir}: {e}")
            return False

    def count(self) -> int:
        """Returns total number of vectors in index."""
        return self.index.ntotal if self.index is not None else 0

    def get_indexed_files(self) -> List[str]:
        """Returns list of distinct source filenames currently indexed."""
        files = set(c["source_file"] for c in self.chunks_metadata if "source_file" in c)
        return sorted(list(files))

    def get_document_stats(self) -> List[Dict[str, Any]]:
        """
        Returns per-document statistics: filename, title, chunk count, page count,
        and a short preview from the first chunk.

        Useful for rendering the interactive sidebar document list.
        """
        stats: Dict[str, Dict[str, Any]] = {}
        for chunk in self.chunks_metadata:
            fname = chunk.get("source_file", "unknown")
            if fname not in stats:
                stats[fname] = {
                    "source_file": fname,
                    "doc_title": chunk.get("doc_title", fname),
                    "chunk_count": 0,
                    "pages": set(),
                    "first_preview": chunk.get("text", "")[:200].strip(),
                }
            stats[fname]["chunk_count"] += 1
            pnum = chunk.get("page_number")
            if pnum is not None:
                stats[fname]["pages"].add(pnum)

        result = []
        for fname, s in sorted(stats.items()):
            result.append({
                "source_file": s["source_file"],
                "doc_title": s["doc_title"],
                "chunk_count": s["chunk_count"],
                "page_count": len(s["pages"]),
                "first_preview": s["first_preview"],
            })
        return result

    def remove_document(self, source_file: str) -> bool:
        """
        Removes all chunks belonging to source_file from the index,
        rebuilds the FAISS index in-place, and persists to disk.

        Returns True if any chunks were removed, False if none matched.
        """
        original_count = len(self.chunks_metadata)
        remaining = [c for c in self.chunks_metadata if c.get("source_file") != source_file]

        if len(remaining) == original_count:
            logger.warning(f"remove_document: No chunks found for '{source_file}'.")
            return False

        removed = original_count - len(remaining)
        logger.info(f"Removing {removed} chunks for '{source_file}'; rebuilding index...")

        if not remaining:
            # Empty index after removal
            self.index = None
            self.chunks_metadata = []
            self.save_local()
            return True

        # Rebuild FAISS from remaining chunks
        texts = [c["text"] for c in remaining]
        embeddings = self.embed_texts(texts)
        dimension = embeddings.shape[1]
        new_index = faiss.IndexFlatIP(dimension)
        new_index.add(embeddings)

        self.index = new_index
        self.chunks_metadata = remaining
        self.save_local()
        logger.info(f"Index rebuilt with {self.index.ntotal} vectors after removal.")
        return True
