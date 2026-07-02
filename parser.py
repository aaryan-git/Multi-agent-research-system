"""PDF parsing utilities for the Reader Agent.

extract_pdf_text() pulls raw text out of a PDF (bytes) using PyMuPDF,
falling back to pdfplumber if PyMuPDF fails or returns nothing usable.

extract_sections() takes that raw text and splits it into common academic
paper sections (Abstract, Introduction, Methods, Results, Conclusion,
References) based on header line matching.
"""

import io
import re

import fitz  # PyMuPDF
import pdfplumber

SECTION_HEADERS = [
    "abstract",
    "introduction",
    "related work",
    "background",
    "methodology",
    "methods",
    "materials and methods",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "references",
]


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract raw text from PDF bytes. Tries PyMuPDF first (fast), falls
    back to pdfplumber (more robust on tricky layouts) if needed."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        if text.strip():
            return text
    except Exception:
        pass

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            if text.strip():
                return text
    except Exception as e:
        return f"Could not extract PDF text: {str(e)}"

    return ""


def extract_sections(text: str) -> dict:
    """Split extracted PDF text into common academic sections by matching
    header lines. Returns an ordered dict of {section_name: content}.
    Returns an empty dict if no recognizable headers are found (caller
    should fall back to using the raw text)."""
    if not text:
        return {}

    pattern = re.compile(
        r"^\s*(" + "|".join(SECTION_HEADERS) + r")\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

    matches = list(pattern.finditer(text))
    if not matches:
        return {}

    sections = {}
    for i, match in enumerate(matches):
        name = match.group(1).strip().title()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            sections[name] = content

    return sections
