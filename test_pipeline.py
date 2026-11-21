from resume_parser import extract_text_from_pdf
from trained_model import analyze_resume


# ============================================================
# RESUME FILE
# ============================================================

PDF_PATH = "resume.pdf"


# ============================================================
# READ RESUME
# ============================================================

print("=" * 80)
print("RESUME ANALYSER")
print("=" * 80)

print("\nReading resume...")

resume_text = extract_text_from_pdf(PDF_PATH)


# ============================================================
# CHECK EXTRACTION
# ============================================================

if not resume_text.strip():

    print("\nERROR: Could not extract text from resume.")

else:

    print("\nResume extracted successfully.")

    print("\nExtracted characters:",
          len(resume_text))


    # ========================================================
    # ANALYZE RESUME
    # ========================================================

    print("\nAnalyzing resume using trained Gemma model...")

    result = analyze_resume(resume_text)


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    print("\n")

    print("=" * 80)
    print("FINAL RESUME ANALYSIS")
    print("=" * 80)

    print(result)

    print("\n")
    print("=" * 80)
    print("ANALYSIS COMPLETED")
    print("=" * 80)