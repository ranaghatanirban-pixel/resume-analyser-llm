from __future__ import annotations

import os
import re
import sys

from resume_parser import extract_text_from_pdf
from trained_model import REQUIRED_SECTIONS, analyze_resume


def test_resume_pipeline(pdf_path: str) -> bool:
    """Run the end-to-end PDF extraction and Gemma LoRA analysis pipeline."""
    print("=" * 75)
    print("        RESUME ANALYSER LLM - FINAL PIPELINE TEST")
    print("=" * 75)

    print("\n" + "-" * 75)
    print("STEP 1: CHECKING RESUME FILE")
    print("-" * 75)

    if not os.path.exists(pdf_path):
        print("\nERROR: Resume PDF not found.")
        print("Expected file:", os.path.abspath(pdf_path))
        return False

    print("\nResume file found:", pdf_path)

    print("\n" + "-" * 75)
    print("STEP 2: EXTRACTING RESUME TEXT")
    print("-" * 75)

    try:
        resume_text = extract_text_from_pdf(pdf_path)
    except Exception as exc:
        print("\nPDF EXTRACTION ERROR:", str(exc))
        return False

    if not resume_text or not resume_text.strip():
        print("\nERROR: No readable text was extracted from the PDF.")
        return False

    print("\nResume text extracted successfully")
    print("Characters extracted:", len(resume_text))

    print("\n" + "-" * 75)
    print("EXTRACTED RESUME TEXT")
    print("-" * 75)
    print(resume_text[:5000])
    if len(resume_text) > 5000:
        print("\n... [Resume text truncated for display] ...")

    print("\n" + "-" * 75)
    print("STEP 3: RUNNING AI MODEL")
    print("-" * 75)
    print("\nLoading/running Gemma + LoRA...")

    try:
        result = analyze_resume(resume_text)
    except Exception as exc:
        print("\nMODEL ERROR:", str(exc))
        return False

    if not result or not result.strip():
        print("\nERROR: The AI model returned an empty result.")
        return False

    print("\nAI analysis generated successfully")

    print("\n" + "-" * 75)
    print("AI ANALYSIS")
    print("-" * 75)
    print(result)

    print("\n" + "-" * 75)
    print("STEP 4: CHECKING REQUIRED SECTIONS")
    print("-" * 75)

    missing_sections: list[str] = []
    for section in REQUIRED_SECTIONS:
        pattern = rf"(?im)^\s*{re.escape(section)}\s*:"
        if re.search(pattern, result):
            print(f"OK: {section}")
        else:
            print(f"MISSING: {section}")
            missing_sections.append(section)

    print("\n" + "-" * 75)
    print("STEP 5: CHECKING DUPLICATE HEADINGS")
    print("-" * 75)

    duplicate_found = False
    for section in REQUIRED_SECTIONS:
        pattern = rf"(?im)^\s*{re.escape(section)}\s*:"
        count = len(re.findall(pattern, result))
        if count > 1:
            duplicate_found = True
            print(f"DUPLICATE: {section} appears {count} times")
        else:
            print(f"OK: {section} appears {count} time")

    print("\n" + "-" * 75)
    print("STEP 6: CHECKING JOB FIT SCORE")
    print("-" * 75)

    score_match = re.search(r"(?im)^\s*Job Fit Score\s*:\s*(\d+)", result)
    score_valid = False
    if score_match:
        score = int(score_match.group(1))
        print(f"\nDetected Job Fit Score: {score}")
        if 0 <= score <= 100:
            print(f"OK: Job Fit Score is valid: {score}/100")
            score_valid = True
        else:
            print(f"INVALID: Job Fit Score out of range: {score}")
    else:
        print("\nINVALID: Job Fit Score could not be detected.")

    print("\n" + "-" * 75)
    print("STEP 7: CHECKING EMPTY SECTIONS")
    print("-" * 75)

    for section in REQUIRED_SECTIONS:
        pattern = (
            rf"(?ims)^\s*{re.escape(section)}\s*:\s*(.*?)"
            rf"(?=^\s*(?:"
            + "|".join(re.escape(s) for s in REQUIRED_SECTIONS)
            + r")\s*:|\Z)"
        )
        match = re.search(pattern, result)
        if match and match.group(1).strip():
            print(f"OK: {section} contains information")
        else:
            print(f"WARN: {section} is empty")

    print("\n" + "-" * 75)
    print("STEP 8: CHECKING FOR CORRUPTED OUTPUT")
    print("-" * 75)

    corrupted_terms = [
        "undefined",
        "null",
        "nan",
        "none none none",
        "candidate candidate candidate",
        "skills skills skills",
        "job fit score job fit score",
    ]
    result_lower = result.lower()
    corrupted_found = [term for term in corrupted_terms if term in result_lower]
    if corrupted_found:
        print("\nWARN: Possible corrupted/repeated terms found:", corrupted_found)
    else:
        print("\nOK: No obvious corrupted output detected.")

    print("\n" + "-" * 75)
    print("STEP 9: CHECKING AI OUTPUT LENGTH")
    print("-" * 75)
    result_length = len(result.strip())
    print("\nAI output characters:", result_length)

    print("\n" + "=" * 75)
    print("                    FINAL TEST RESULT")
    print("=" * 75)

    test_passed = (
        not missing_sections
        and not duplicate_found
        and score_valid
        and not corrupted_found
    )

    if test_passed:
        print("\nPIPELINE TEST PASSED")
    else:
        print("\nPIPELINE TEST COMPLETED WITH WARNINGS")

    print("\n" + "=" * 75)
    return test_passed


if __name__ == "__main__":
    target_pdf = sys.argv[1] if len(sys.argv) > 1 else "resume.pdf"
    raise SystemExit(0 if test_resume_pipeline(target_pdf) else 1)
