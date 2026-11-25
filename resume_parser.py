from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Union

import pymupdf


def remove_duplicate_lines(text: str) -> str:
    """Remove duplicate lines while preserving the original order."""
    cleaned_lines: list[str] = []
    seen_lines: set[str] = set()

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        normalized = re.sub(r"\s+", " ", line)
        comparison = normalized.lower()

        if comparison not in seen_lines:
            seen_lines.add(comparison)
            cleaned_lines.append(normalized)

    return "\n".join(cleaned_lines)


def remove_repeated_text(text: str) -> str:
    """Remove consecutive duplicate lines caused by overlapping PDF text layers."""
    cleaned_lines: list[str] = []
    previous_line: str | None = None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        lowered = line.lower()
        if lowered == previous_line:
            continue

        cleaned_lines.append(line)
        previous_line = lowered

    return "\n".join(cleaned_lines)


def extract_text_from_pdf(pdf_source: Union[str, os.PathLike[str], bytes]) -> str:
    """Extract and clean text from a PDF file path or raw PDF bytes."""
    pages_text: list[str] = []

    try:
        if isinstance(pdf_source, (bytes, bytearray)):
            open_kwargs = {"stream": bytes(pdf_source), "filetype": "pdf"}
        else:
            open_kwargs = {"filename": os.fspath(pdf_source)}

        with pymupdf.open(**open_kwargs) as document:
            for page in document:
                page_text = page.get_text("text")
                if page_text:
                    pages_text.append(page_text)
    except Exception:
        return ""

    raw_text = "\n".join(pages_text)
    if not raw_text.strip():
        return ""

    cleaned = remove_repeated_text(raw_text)
    cleaned = remove_duplicate_lines(cleaned)

    final_lines = [
        re.sub(r"\s+", " ", line.strip())
        for line in cleaned.splitlines()
        if line.strip()
    ]
    return "\n".join(final_lines)


if __name__ == "__main__":
    default_pdf = Path("resume.pdf")
    print("=" * 80)
    print("RESUME PDF PARSER")
    print("=" * 80)
    print("\nReading resume...")

    extracted = extract_text_from_pdf(default_pdf)
    if not extracted.strip():
        print("\nERROR: Could not extract text from PDF.")
    else:
        print("\nResume extracted successfully!")
        print("\nExtracted characters:", len(extracted))
        print("\n" + "=" * 80)
        print("EXTRACTED RESUME TEXT")
        print("=" * 80)
        print(extracted)
        print("\n" + "=" * 80)
        print("PARSING COMPLETED")
        print("=" * 80)
