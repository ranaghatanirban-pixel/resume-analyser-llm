import fitz
import re


# ============================================================
# REMOVE DUPLICATE LINES
# ============================================================

def remove_duplicate_lines(text):
    """
    Remove duplicate lines while preserving the original order.
    """

    lines = text.splitlines()

    cleaned_lines = []
    seen_lines = set()

    for line in lines:

        line = line.strip()

        # Skip completely empty lines
        if not line:
            continue

        # Normalize spaces
        normalized = re.sub(r"\s+", " ", line)

        # Create lowercase comparison version
        comparison = normalized.lower()

        # Remove duplicate line
        if comparison not in seen_lines:

            seen_lines.add(comparison)
            cleaned_lines.append(normalized)

    return "\n".join(cleaned_lines)


# ============================================================
# REMOVE REPEATED CONSECUTIVE TEXT
# ============================================================

def remove_repeated_text(text):
    """
    Remove repeated text fragments that may occur because
    of overlapping PDF text layers.
    """

    lines = text.splitlines()

    cleaned_lines = []

    previous_line = None

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # If the same line appears consecutively, ignore it
        if line.lower() == previous_line:

            continue

        cleaned_lines.append(line)

        previous_line = line.lower()

    return "\n".join(cleaned_lines)


# ============================================================
# EXTRACT TEXT FROM PDF
# ============================================================

def extract_text_from_pdf(pdf_path):

    text = ""

    try:

        document = fitz.open(pdf_path)

        print("Number of pages:", len(document))

        for page_number, page in enumerate(document):

            print(
                f"Reading page {page_number + 1}..."
            )

            # Extract text from page
            page_text = page.get_text(
                "text"
            )

            text += page_text

            text += "\n"


        document.close()


    except Exception as e:

        print(
            f"Error reading PDF: {e}"
        )

        return ""


    # ========================================================
    # CLEAN EXTRACTED TEXT
    # ========================================================

    text = remove_repeated_text(text)

    text = remove_duplicate_lines(text)


    # ========================================================
    # NORMALIZE SPACES
    # ========================================================

    lines = text.splitlines()

    final_lines = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # Replace multiple spaces with one
        line = re.sub(
            r"\s+",
            " ",
            line
        )

        final_lines.append(line)


    text = "\n".join(final_lines)


    return text


# ============================================================
# TEST PDF PARSER
# ============================================================

if __name__ == "__main__":

    pdf_path = "resume.pdf"

    print("=" * 80)
    print("RESUME PDF PARSER")
    print("=" * 80)

    print("\nReading resume...")

    resume_text = extract_text_from_pdf(
        pdf_path
    )


    if not resume_text.strip():

        print(
            "\nERROR: Could not extract text from PDF."
        )

    else:

        print(
            "\nResume extracted successfully!"
        )

        print(
            "\nExtracted characters:",
            len(resume_text)
        )

        print(
            "\n"
            + "=" * 80
        )

        print(
            "EXTRACTED RESUME TEXT"
        )

        print(
            "=" * 80
        )

        print(resume_text)

        print(
            "\n"
            + "=" * 80
        )

        print(
            "PARSING COMPLETED"
        )

        print(
            "=" * 80
        )