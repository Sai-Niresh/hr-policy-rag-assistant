"""
PDF Ingestion Module for HR Policy Documents.

Extracts text from PDF documents page-by-page using PyMuPDF (fitz),
cleans whitespace, preserves document metadata, and provides graceful
error handling for corrupted or scanned files.
"""

import json
import logging
import os
import re
import urllib.request
from typing import Any, Dict, List, Optional

try:
    import pymupdf as fitz
except ImportError:
    import fitz

logger = logging.getLogger(__name__)


def clean_text(raw_text: str) -> str:
    """
    Cleans raw extracted text by normalizing whitespace while preserving
    paragraph structure, punctuation, and semantic meaning.
    """
    if not raw_text:
        return ""

    # Replace form feed and other unprintable control characters
    text = raw_text.replace("\x0c", "\n")

    # Replace carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Replace multiple inline spaces/tabs with a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize more than two consecutive line breaks into two (preserve paragraphs)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip line-by-line trailing/leading spaces
    lines = [line.strip() for line in text.split("\n")]
    cleaned = "\n".join(lines).strip()

    return cleaned


def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text page by page from a given PDF file path.

    Returns a list of dicts:
        [
            {
                "page_number": int (1-indexed),
                "text": str (cleaned page content),
                "char_count": int
            },
            ...
        ]

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid PDF or has no extractable text.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Failed to open PDF {pdf_path}: {e}")
        raise ValueError(f"Could not open or parse PDF: {pdf_path}. Error: {e}")

    pages: List[Dict[str, Any]] = []
    total_chars = 0

    try:
        for page_index in range(len(doc)):
            page_num = page_index + 1
            page = doc[page_index]
            raw_text = page.get_text("text") or ""
            cleaned = clean_text(raw_text)

            pages.append({
                "page_number": page_num,
                "text": cleaned,
                "char_count": len(cleaned)
            })
            total_chars += len(cleaned)

    finally:
        doc.close()

    if total_chars == 0:
        logger.warning(
            f"PDF '{os.path.basename(pdf_path)}' produced 0 characters of text. "
            "It may be a scanned image-only PDF without OCR."
        )

    return pages


def validate_pdf_file(
    file_bytes: bytes,
    filename: str,
    max_size_mb: float = 20.0
) -> Dict[str, Any]:
    """
    Validates an uploaded PDF file before indexing.

    Checks:
    - File size within max_size_mb limit
    - Valid PDF header signature (starts with %PDF-)
    - Parseable by PyMuPDF without corruption errors
    - Contains extractable text (not a scanned image-only file)

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename: Original filename (used in error messages).
        max_size_mb: Maximum allowed size in megabytes (default 20 MB).

    Returns:
        A dict with keys:
            ok (bool): True if the file is valid, False otherwise.
            error_type (str | None): One of 'size', 'format', 'corrupt', 'scanned', or None.
            message (str): Human-readable description of the validation result.
            page_count (int): Number of pages (0 if validation failed before parsing).
            total_chars (int): Total character count across all pages (0 if failed).
    """
    result: Dict[str, Any] = {
        "ok": False,
        "error_type": None,
        "message": "",
        "page_count": 0,
        "total_chars": 0,
    }

    # --- Size check ---
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        result["error_type"] = "size"
        result["message"] = (
            f"File '{filename}' is {size_mb:.1f} MB, which exceeds the {max_size_mb:.0f} MB limit."
        )
        return result

    # --- PDF signature check ---
    if not file_bytes.startswith(b"%PDF-"):
        result["error_type"] = "format"
        result["message"] = (
            f"'{filename}' does not appear to be a valid PDF file (missing PDF header)."
        )
        return result

    # --- PyMuPDF parse + text extraction check ---
    import io
    try:
        doc = fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf")
    except Exception as e:
        result["error_type"] = "corrupt"
        result["message"] = f"'{filename}' could not be opened — it may be corrupted or password-protected. ({e})"
        return result

    total_chars = 0
    page_count = 0
    try:
        page_count = len(doc)
        for page in doc:
            raw = page.get_text("text") or ""
            total_chars += len(clean_text(raw))
    finally:
        doc.close()

    result["page_count"] = page_count
    result["total_chars"] = total_chars

    if total_chars == 0:
        result["error_type"] = "scanned"
        result["message"] = (
            f"'{filename}' contains no extractable text. "
            "It may be a scanned image-only PDF. Please use a text-searchable PDF."
        )
        return result

    result["ok"] = True
    result["message"] = (
        f"'{filename}' is valid: {page_count} page(s), {total_chars:,} characters."
    )
    return result




def download_default_policies(
    manifest_path: str = "data/manifest.json",
    target_dir: str = "data/policies"
) -> List[str]:
    """
    Downloads default policy documents specified in manifest.json if they
    do not already exist in target_dir.

    Returns list of downloaded or existing PDF file paths.
    """
    os.makedirs(target_dir, exist_ok=True)

    if not os.path.exists(manifest_path):
        logger.warning(f"Manifest not found at {manifest_path}; skipping default download.")
        return []

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    downloaded_paths = []

    for item in manifest.get("policies", []):
        filename = item["filename"]
        url = item["original_url"]
        dest_path = os.path.join(target_dir, filename)

        if not os.path.exists(dest_path) or os.path.getsize(dest_path) == 0:
            logger.info(f"Downloading {filename} from {url}...")
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) HR-Policy-RAG-Assistant/1.0"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp, open(dest_path, "wb") as out_f:
                out_f.write(resp.read())
            logger.info(f"Downloaded {filename} ({os.path.getsize(dest_path)} bytes)")
        else:
            logger.info(f"Using existing policy file: {dest_path}")

        downloaded_paths.append(dest_path)

    return downloaded_paths


def load_all_policies(
    policies_dir: str = "data/policies",
    manifest_path: Optional[str] = "data/manifest.json"
) -> List[Dict[str, Any]]:
    """
    Ingests all PDF files present in policies_dir, associating each with
    manifest metadata if present.

    Returns a list of structured document dictionaries:
    [
        {
            "filename": str,
            "filepath": str,
            "title": str,
            "pages": List[Dict[str, Any]],
            "total_pages": int,
            "is_scanned_or_empty": bool
        },
        ...
    ]
    """
    manifest_meta = {}
    if manifest_path and os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
                for item in manifest_data.get("policies", []):
                    manifest_meta[item["filename"]] = item
        except Exception as e:
            logger.warning(f"Failed to load manifest at {manifest_path}: {e}")

    if not os.path.exists(policies_dir):
        return []

    pdf_files = [f for f in os.listdir(policies_dir) if f.lower().endswith(".pdf")]
    documents = []

    for fname in sorted(pdf_files):
        fpath = os.path.join(policies_dir, fname)
        meta = manifest_meta.get(fname, {})
        title = meta.get("title", fname.replace("_", " ").replace(".pdf", "").title())

        try:
            pages = extract_text_from_pdf(fpath)
            total_chars = sum(p["char_count"] for p in pages)
            documents.append({
                "filename": fname,
                "filepath": fpath,
                "title": title,
                "pages": pages,
                "total_pages": len(pages),
                "is_scanned_or_empty": (total_chars == 0)
            })
        except Exception as e:
            logger.error(f"Error processing policy PDF {fpath}: {e}")

    return documents
