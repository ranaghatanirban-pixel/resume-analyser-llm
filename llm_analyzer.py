import ollama
from resume_parser import extract_text_from_pdf


def analyze_resume(resume_text):
    """
    Analyze resume text using an Ollama LLM.

    Args:
        resume_text: Extracted text from the resume

    Returns:
        LLM-generated resume analysis
    """

    prompt = f"""
You are an expert resume analyzer and career assistant.

Analyze the following resume.

Provide the analysis in the following sections:

1. Candidate Summary
2. Skills
3. Education
4. Work Experience
5. Strengths
6. Weaknesses
7. Missing or Recommended Skills
8. Resume Improvement Suggestions
9. Suitable Job Roles

Resume:
-------------------------
{resume_text}
-------------------------

Give a clear and useful analysis.
"""

    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]

    except Exception as e:
        return f"Error communicating with Ollama: {e}"


if __name__ == "__main__":

    pdf_path = "resume.pdf"

    resume_text = extract_text_from_pdf(pdf_path)

    if not resume_text.strip():
        print("Could not extract text from the resume.")
    else:
        result = analyze_resume(resume_text)

        print("\n========== RESUME ANALYSIS ==========\n")
        print(result)