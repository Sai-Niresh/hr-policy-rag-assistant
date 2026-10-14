"""
HR Policy RAG Assistant - Streamlit Web Application.

A complete Retrieval-Augmented Generation (RAG) assistant for querying
HR policy documents with verifiable citations and local vector search.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# ----- Reliable .env loading (must be BEFORE any src imports that read env) -----
_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=True)

from src.ingest import (  # noqa: E402
    download_default_policies,
    extract_text_from_pdf,
    load_all_policies,
    validate_pdf_file,
)
from src.chunking import chunk_all_documents, chunk_document  # noqa: E402
from src.vector_store import FAISSVectorStore  # noqa: E402
from src.answer_generator import RAGAnswerGenerator, ABSTENTION_MESSAGE  # noqa: E402

# ──────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="HR Policy RAG Assistant",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .citation-card {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 4px;
    }
    .citation-title {
        font-weight: 600;
        color: #1E40AF;
        font-size: 0.95rem;
    }
    .citation-text {
        font-size: 0.88rem;
        color: #334155;
        margin-top: 4px;
        font-style: italic;
    }
    .status-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-success { background-color: #DCFCE7; color: #166534; }
    .badge-warning { background-color: #FEF9C3; color: #854D0E; }
    .doc-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .doc-card-title {
        font-weight: 600;
        font-size: 0.9rem;
        color: #1E293B;
    }
    .doc-card-meta {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 3px;
    }
    .doc-preview {
        font-size: 0.8rem;
        color: #475569;
        font-style: italic;
        margin-top: 6px;
        border-top: 1px solid #E2E8F0;
        padding-top: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def _load_manifest() -> dict:
    """Returns the manifest dict, keyed by filename."""
    manifest_path = _PROJECT_ROOT / "data" / "manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {item["filename"]: item for item in data.get("policies", [])}
        except Exception:
            pass
    return {}


def _file_content_hash(file_bytes: bytes) -> str:
    """Returns a short SHA-256 hex hash of file bytes for duplicate detection."""
    return hashlib.sha256(file_bytes).hexdigest()[:12]


def _get_indexed_hashes() -> set:
    """
    Reads hash tags stored in the session state (persisted only for this session).
    """
    return st.session_state.get("indexed_hashes", set())


# ──────────────────────────────────────────────
# Initialization (cached)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def initialize_vector_store() -> FAISSVectorStore:
    """
    Loads existing FAISS index from disk. If none exists, downloads default
    policy PDFs and builds the index automatically.
    """
    store = FAISSVectorStore(index_dir=str(_PROJECT_ROOT / "data" / "faiss_index"))
    loaded = store.load_local()

    if not loaded or store.count() == 0:
        download_default_policies(
            manifest_path=str(_PROJECT_ROOT / "data" / "manifest.json"),
            target_dir=str(_PROJECT_ROOT / "data" / "policies"),
        )
        documents = load_all_policies(
            policies_dir=str(_PROJECT_ROOT / "data" / "policies"),
            manifest_path=str(_PROJECT_ROOT / "data" / "manifest.json"),
        )
        chunks = chunk_all_documents(documents)
        store.build_index(chunks)

    return store


def rebuild_index(vector_store: FAISSVectorStore) -> None:
    """Re-indexes all policy files currently in data/policies/."""
    documents = load_all_policies(
        policies_dir=str(_PROJECT_ROOT / "data" / "policies"),
        manifest_path=str(_PROJECT_ROOT / "data" / "manifest.json"),
    )
    chunks = chunk_all_documents(documents)
    vector_store.build_index(chunks)


# ──────────────────────────────────────────────
# Init objects
# ──────────────────────────────────────────────
vector_store = initialize_vector_store()
api_key = os.getenv("GROQ_API_KEY", "")
answer_generator = RAGAnswerGenerator(vector_store=vector_store, api_key=api_key)

# Session state for removal confirmation and indexed hash tracking
if "confirm_remove" not in st.session_state:
    st.session_state.confirm_remove = None
if "indexed_hashes" not in st.session_state:
    st.session_state.indexed_hashes = set()

manifest = _load_manifest()

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Policy Knowledge Base")

    # ── API Status ──────────────────────────────
    is_api_ready = answer_generator.is_api_key_configured()
    if is_api_ready:
        st.markdown(
            '<span class="status-badge badge-success">✓ Groq API Connected</span>',
            unsafe_allow_html=True,
        )
        model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        st.caption(f"Model: `{model_name}`")
    else:
        st.markdown(
            '<span class="status-badge badge-warning">⚠ API Key Missing</span>',
            unsafe_allow_html=True,
        )
        st.caption("Add `GROQ_API_KEY` to `.env` to enable live LLM answers.")

    st.divider()

    # ── Index Summary Metrics ────────────────────
    total_vectors = vector_store.count()
    doc_stats = vector_store.get_document_stats()

    col_a, col_b = st.columns(2)
    col_a.metric("Policy Chunks", total_vectors)
    col_b.metric("Documents", len(doc_stats))

    st.divider()

    # ── Interactive Document Cards ───────────────
    st.subheader("📚 Active Policy Documents")

    if not doc_stats:
        st.info("No policy documents are indexed yet. Upload a PDF below.")
    else:
        for doc in doc_stats:
            fname = doc["source_file"]
            title = doc["doc_title"]
            chunk_count = doc["chunk_count"]
            page_count = doc["page_count"]
            preview = doc["first_preview"]

            # Look up manifest for publisher URL
            meta = manifest.get(fname, {})
            publisher = meta.get("publisher", "")
            url = meta.get("original_url", "")

            # One expander per document
            with st.expander(f"📄 {title}", expanded=False):
                st.markdown(f"**File:** `{fname}`")
                st.markdown(f"**Pages indexed:** {page_count} &nbsp;|&nbsp; **Chunks:** {chunk_count}")
                if publisher:
                    st.markdown(f"**Publisher:** {publisher}")
                if url:
                    st.markdown(f"**Source:** [{url[:55]}...]({url})" if len(url) > 58 else f"**Source:** [{url}]({url})")

                st.markdown(
                    f'<div class="doc-preview">"{preview}..."</div>',
                    unsafe_allow_html=True,
                )

                # ── Remove document ────────────────────────
                btn_key = f"remove_{fname}"
                confirm_key = f"confirm_{fname}"

                if st.session_state.confirm_remove == fname:
                    st.warning(f"⚠️ Remove **{fname}** from the index?")
                    c1, c2 = st.columns(2)
                    if c1.button("Yes, Remove", key=f"yes_{fname}", use_container_width=True):
                        with st.spinner("Removing and rebuilding index..."):
                            removed = vector_store.remove_document(fname)
                            # Also delete physical file if present
                            pdf_path = _PROJECT_ROOT / "data" / "policies" / fname
                            if pdf_path.exists():
                                pdf_path.unlink()
                        st.session_state.confirm_remove = None
                        initialize_vector_store.clear()
                        st.rerun()
                    if c2.button("Cancel", key=f"no_{fname}", use_container_width=True):
                        st.session_state.confirm_remove = None
                        st.rerun()
                else:
                    if st.button(
                        "🗑️ Remove from Knowledge Base",
                        key=btn_key,
                        use_container_width=True,
                    ):
                        st.session_state.confirm_remove = fname
                        st.rerun()

    st.divider()

    # ── Upload New Policy PDF ────────────────────
    st.subheader("📤 Add New Policy PDF")
    st.caption(
        "Upload a text-searchable HR policy PDF (max 20 MB). "
        "The document will be validated, chunked, and added to the vector index."
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF file:",
        type=["pdf"],
        help="Only text-searchable PDFs are supported. Scanned images require OCR.",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()

        # ── Validate ──────────────────────────────
        validation = validate_pdf_file(file_bytes, uploaded_file.name)

        if not validation["ok"]:
            err_type = validation["error_type"]
            icons = {"size": "📦", "format": "🚫", "corrupt": "⚠️", "scanned": "🔍"}
            st.error(f"{icons.get(err_type, '❌')} {validation['message']}")
        else:
            st.success(
                f"✅ {validation['page_count']} page(s) | "
                f"{validation['total_chars']:,} chars — ready to index"
            )

            # ── Duplicate check ───────────────────
            content_hash = _file_content_hash(file_bytes)
            already_indexed_name = uploaded_file.name in vector_store.get_indexed_files()
            already_indexed_hash = content_hash in st.session_state.indexed_hashes

            if already_indexed_name or already_indexed_hash:
                st.warning(
                    f"⚠️ **'{uploaded_file.name}'** is already in the knowledge base. "
                    "To replace it, remove it first using the document card above."
                )
            else:
                if st.button(
                    "⚡ Index This Policy Document",
                    use_container_width=True,
                    type="primary",
                ):
                    with st.spinner("Extracting text, creating embeddings, and updating index..."):
                        # Save to policies dir
                        save_dir = _PROJECT_ROOT / "data" / "policies"
                        save_dir.mkdir(parents=True, exist_ok=True)
                        dest_path = save_dir / uploaded_file.name

                        dest_path.write_bytes(file_bytes)

                        try:
                            pages = extract_text_from_pdf(str(dest_path))
                            doc_dict = {
                                "filename": uploaded_file.name,
                                "filepath": str(dest_path),
                                "title": (
                                    uploaded_file.name
                                    .replace("_", " ")
                                    .replace(".pdf", "")
                                    .title()
                                ),
                                "pages": pages,
                                "total_pages": len(pages),
                                "is_scanned_or_empty": validation["total_chars"] == 0,
                            }
                            new_chunks = chunk_document(doc_dict)
                            vector_store.add_chunks(new_chunks)
                            st.session_state.indexed_hashes.add(content_hash)

                            st.success(
                                f"✅ **'{uploaded_file.name}'** indexed! "
                                f"Added **{len(new_chunks)} chunks** to the knowledge base."
                            )
                            initialize_vector_store.clear()
                            st.rerun()
                        except Exception as e:
                            # Clean up the file if indexing failed
                            if dest_path.exists():
                                dest_path.unlink()
                            st.error(f"❌ Failed to index '{uploaded_file.name}': {e}")

    # ── Rebuild index ────────────────────────────
    st.divider()
    if st.button("🔄 Rebuild Entire Index from Scratch", use_container_width=True):
        with st.spinner("Re-indexing all policy files in data/policies/..."):
            rebuild_index(vector_store)
            initialize_vector_store.clear()
            st.success("✅ Vector store index rebuilt successfully!")
            st.rerun()


# ──────────────────────────────────────────────
# MAIN AREA
# ──────────────────────────────────────────────
st.markdown('<div class="main-title">HR Policy RAG Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    "Ask questions about Dickinson College HR policies (Remote Work, Vacation, Sick Leave Bank). "
    "Answers are strictly grounded in retrieved policy passages with verifiable source citations."
    "</div>",
    unsafe_allow_html=True,
)

# ── How It Works ─────────────────────────────
with st.expander("ℹ️ How This Assistant Works (RAG Architecture)"):
    st.markdown("""
    1. **Local Text Extraction**: PyMuPDF (`fitz`) extracts clean text page-by-page from HR policy PDFs, preserving exact page numbers.
    2. **Context-Aware Chunking**: Documents are split into ~800 character chunks (with 120 character overlap) to balance context coherence with retrieval precision.
    3. **Local Dense Embeddings**: Chunks are embedded on CPU using `sentence-transformers/all-MiniLM-L6-v2` into 384-dimensional dense vectors.
    4. **Vector Retrieval**: FAISS CPU conducts cosine similarity search to retrieve the top 4 most relevant chunks for your query.
    5. **Grounded Answer Generation**: An LLM (via Groq API) synthesizes an answer using **strictly** the retrieved context, citing document names and page numbers.
    6. **Hallucination Guard**: If the question cannot be answered from the policy documents, the assistant explicitly abstains rather than inventing answers.
    """)

# ── Sample Questions ──────────────────────────
sample_questions = [
    "-- Select an example question or type below --",
    "What is the maximum paid sick leave an employee can receive from the Sick Leave Bank?",
    "What requirements must an employee meet to withdraw hours from the Sick Leave Bank?",
    "How many days of vacation do full-time nonexempt employees earn compared to exempt staff?",
    "Who must approve an employee's remote work arrangement?",
    "What is the reimbursement policy for pet health insurance?",
]

selected_sample = st.selectbox("💡 Example Questions:", sample_questions)

# ── Query Form ────────────────────────────────
with st.form("query_form"):
    default_text = "" if selected_sample.startswith("--") else selected_sample
    user_query = st.text_input(
        "Enter your HR policy question:",
        value=default_text,
        placeholder="e.g. Can an employee roll over unused vacation days into the next fiscal year?",
    )
    col1, col2 = st.columns([1, 5])
    with col1:
        submit_btn = st.form_submit_button("Ask Question", use_container_width=True)
    with col2:
        top_k_val = st.slider("Retrieval Chunks (Top K)", min_value=1, max_value=8, value=4)

if submit_btn and user_query:
    with st.spinner("Searching HR policy index and generating grounded answer..."):
        result = answer_generator.answer_question(question=user_query, top_k=top_k_val)

    st.subheader("📝 Answer")
    if result["status"] == "missing_api_key":
        st.warning(result["answer"])
    elif result["status"] == "error":
        st.error(result["answer"])
    else:
        st.markdown(result["answer"])

    # ── Citations ─────────────────────────────
    citations = result.get("citations", [])
    if citations:
        st.subheader("📑 Source Citations")
        for cit in citations:
            st.markdown(f"""
            <div class="citation-card">
                <div class="citation-title">📌 {cit['doc_title']} — Page {cit['page_number']}</div>
                <div class="citation-text">"{cit['excerpt']}"</div>
                <div style="font-size: 0.78rem; color: #64748B; margin-top: 4px;">
                    Source: <code>{cit['source_file']}</code> | Similarity Score: {cit['similarity_score']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Retrieved Chunks Inspector ────────────
    retrieved_chunks = result.get("retrieved_chunks", [])
    if retrieved_chunks:
        with st.expander(f"🔍 Inspect Retrieved Context Chunks ({len(retrieved_chunks)} passages)"):
            for i, chunk in enumerate(retrieved_chunks, start=1):
                st.markdown(f"**Chunk #{i} | {chunk['doc_title']} (Page {chunk['page_number']})**")
                st.caption(f"Chunk ID: `{chunk['chunk_id']}` | Cosine Similarity: `{chunk['similarity_score']}`")
                st.text_area(
                    label=f"Content #{i}",
                    value=chunk["text"],
                    height=120,
                    key=f"chunk_viewer_{i}_{chunk['chunk_id']}",
                    disabled=True,
                )
                st.divider()

elif submit_btn and not user_query:
    st.warning("Please enter a question before clicking Ask.")
