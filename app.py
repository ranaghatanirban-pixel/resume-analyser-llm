import streamlit as st
import tempfile
import os

from resume_parser import extract_text_from_pdf
from trained_model import analyze_resume


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Resume Analyser LLM",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📄 Resume Analyser LLM")

st.write(
    "Upload your resume PDF and get an AI-powered "
    "professional resume analysis using a fine-tuned "
    "Gemma + LoRA model."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📋 Resume Analyser")

st.sidebar.write(
    """
This application analyzes a resume and provides:

- Candidate Name
- Target Field
- Education
- Experience
- Skills
- Strengths
- Missing Skills
- Job Fit Score
"""
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your Resume (PDF)",
    type=["pdf"]
)


# ============================================================
# PROCESS UPLOADED FILE
# ============================================================

if uploaded_file is not None:

    st.success("Resume uploaded successfully! ✅")

    st.write("### 📄 Uploaded File")

    st.write(uploaded_file.name)


    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    if st.button(
        "🔍 Analyze Resume",
        type="primary"
    ):

        # ----------------------------------------------------
        # Create temporary PDF
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(
                uploaded_file.getbuffer()
            )

            temp_pdf_path = temp_file.name


        try:

            # =================================================
            # STEP 1 — EXTRACT TEXT
            # =================================================

            with st.spinner(
                "📖 Reading resume..."
            ):

                resume_text = extract_text_from_pdf(
                    temp_pdf_path
                )


            # =================================================
            # CHECK EXTRACTION
            # =================================================

            if not resume_text.strip():

                st.error(
                    "❌ Could not extract text from "
                    "the uploaded PDF."
                )

            else:

                st.success(
                    "Resume text extracted successfully! ✅"
                )


                # =============================================
                # SHOW EXTRACTED TEXT
                # =============================================

                with st.expander(
                    "📄 View Extracted Resume Text"
                ):

                    st.text(
                        resume_text
                    )


                # =============================================
                # STEP 2 — AI ANALYSIS
                # =============================================

                with st.spinner(
                    "🤖 Analyzing resume using "
                    "Gemma + LoRA..."
                ):

                    result = analyze_resume(
                        resume_text
                    )


                # =============================================
                # ANALYSIS COMPLETE
                # =============================================

                st.success(
                    "Resume analysis completed successfully! ✅"
                )


                # =============================================
                # DISPLAY RESULT
                # =============================================

                st.header(
                    "📊 Resume Analysis"
                )

                st.markdown(
                    result
                )


                # =============================================
                # DOWNLOAD ANALYSIS
                # =============================================

                st.download_button(
                    label="⬇️ Download Analysis",
                    data=result,
                    file_name="resume_analysis.txt",
                    mime="text/plain"
                )


        except Exception as e:

            st.error(
                f"❌ An error occurred: {e}"
            )


        finally:

            # =============================================
            # DELETE TEMPORARY PDF
            # =============================================

            if os.path.exists(
                temp_pdf_path
            ):

                os.remove(
                    temp_pdf_path
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Built using Streamlit, PyMuPDF, Gemma 2B and LoRA."
)