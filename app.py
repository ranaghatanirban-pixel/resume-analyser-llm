from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st

from resume_parser import extract_text_from_pdf
from trained_model import (
    analyze_resume,
    extract_job_fit_score,
    load_model_and_tokenizer,
    parse_analysis,
)

st.set_page_config(
    page_title="Resume Analyser LLM",
    page_icon=":material/description:",
    layout="wide",
)

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_resumes"


@st.cache_resource(show_spinner=False)
def get_cached_model():
    """Load and cache the Gemma + LoRA model across Streamlit reruns."""
    return load_model_and_tokenizer()


@st.cache_data(max_entries=32, show_spinner=False)
def extract_resume_text_cached(pdf_bytes: bytes) -> str:
    """Cache PDF text extraction keyed by file bytes."""
    return extract_text_from_pdf(pdf_bytes)


def split_items(text: str) -> list[str]:
    """Split a comma- or bullet-separated field into clean items."""
    if not text or text.strip().lower() == "not mentioned":
        return []
    return [item.strip(" •-*") for item in text.split(",") if item.strip(" •-*")]


def build_score_gauge_spec(score: int) -> dict:
    """Build an ECharts gauge configuration for the job fit score."""
    if score >= 75:
        accent = "#22C55E"
    elif score >= 50:
        accent = "#F59E0B"
    else:
        accent = "#EF4444"

    return {
        "series": [
            {
                "type": "gauge",
                "startAngle": 200,
                "endAngle": -20,
                "min": 0,
                "max": 100,
                "splitNumber": 5,
                "progress": {"show": True, "width": 14, "itemStyle": {"color": accent}},
                "pointer": {"show": False},
                "axisLine": {"lineStyle": {"width": 14}},
                "axisTick": {"show": False},
                "splitLine": {"show": False},
                "axisLabel": {"show": False},
                "anchor": {"show": False},
                "title": {"show": True, "offsetCenter": [0, "65%"], "fontSize": 13},
                "detail": {
                    "valueAnimation": True,
                    "offsetCenter": [0, "5%"],
                    "fontSize": 28,
                    "fontWeight": "bold",
                    "formatter": "{value}/100",
                },
                "data": [{"value": score, "name": "Job fit score"}],
            }
        ]
    }


def init_session_state() -> None:
    """Initialize session state defaults in one place."""
    st.session_state.setdefault("analysis_cache", {})
    st.session_state.setdefault("active_resume_key", None)


init_session_state()

sample_files = (
    sorted(SAMPLE_DIR.glob("*.pdf")) if SAMPLE_DIR.is_dir() else []
)
sample_options = {"None (upload your own PDF)": None}
for sample_path in sample_files:
    label = sample_path.stem.replace("_", " ").title()
    sample_options[label] = sample_path

with st.sidebar:
    st.header("ResumeAI", icon=":material/description:")
    st.caption("Structured resume analysis powered by Gemma 2B + LoRA.")

    selected_sample_label = st.selectbox(
        "Sample resume",
        options=list(sample_options.keys()),
        key="sample_resume_choice",
        help="Pick a bundled sample PDF or leave unselected to upload your own.",
    )

    st.subheader("Capabilities")
    st.markdown(
        "- Candidate & target role extraction\n"
        "- Education & experience summary\n"
        "- Skills & missing skills detection\n"
        "- Job fit score (0–100)\n"
        "- Downloadable plain-text report"
    )

    st.subheader("Model stack")
    st.caption("Gemma 2B Instruct • PEFT LoRA • PyMuPDF • Streamlit")

st.title("AI resume analyser", icon=":material/fact_check:")
st.caption(
    "Upload a resume PDF or pick a sample resume from the sidebar to extract text and generate a structured candidate assessment."
)

uploaded_file = st.file_uploader(
    "Upload resume PDF",
    type=["pdf"],
    key="resume_uploader",
    help="Select a text-based PDF resume.",
)

pdf_bytes: bytes | None = None
file_name: str | None = None

if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name
elif sample_options[selected_sample_label] is not None:
    sample_path = sample_options[selected_sample_label]
    if sample_path is not None and sample_path.is_file():
        pdf_bytes = sample_path.read_bytes()
        file_name = sample_path.name

if pdf_bytes is not None and file_name is not None:
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()
    resume_key = f"{file_name}:{file_hash}"
    file_size_kb = len(pdf_bytes) / 1024

    with st.container(border=True):
        info_col, action_col = st.columns([3, 1], vertical_alignment="center")
        with info_col:
            st.markdown(f"**{file_name}**")
            st.caption(f"Ready for analysis • {file_size_kb:.1f} KB")
        with action_col:
            analyze_clicked = st.button(
                "Analyse resume",
                key="analyze_btn",
                type="primary",
                icon=":material/analytics:",
                width="stretch",
            )

    if analyze_clicked:
        with st.status("Analysing resume...", expanded=True) as status:
            st.write("Extracting text from PDF...")
            resume_text = extract_resume_text_cached(pdf_bytes)

            if not resume_text.strip():
                status.update(label="Could not extract readable text", state="error")
                st.error(
                    "The PDF does not contain selectable text. Scanned or image-only PDFs require OCR.",
                    title="Unreadable PDF",
                    icon=":material/error:",
                )
            else:
                st.write("Loading Gemma 2B + LoRA model...")
                get_cached_model()

                st.write("Generating structured analysis...")
                raw_result = analyze_resume(resume_text)
                sections = parse_analysis(raw_result)

                st.session_state["analysis_cache"][resume_key] = {
                    "resume_text": resume_text,
                    "raw_result": raw_result,
                    "sections": sections,
                }
                st.session_state["active_resume_key"] = resume_key
                status.update(label="Analysis complete", state="complete")

    cached_entry = st.session_state["analysis_cache"].get(resume_key)
    if cached_entry is not None:
        resume_text = cached_entry["resume_text"]
        raw_result = cached_entry["raw_result"]
        sections = cached_entry["sections"]
        score = extract_job_fit_score(sections.get("Job Fit Score", "0"))

        st.subheader("Assessment overview", icon=":material/insights:")

        metric_col1, metric_col2, gauge_col = st.columns([1, 1, 1.2])
        with metric_col1:
            st.metric(
                "Candidate",
                sections.get("Candidate Name", "Not mentioned"),
                border=True,
                height="stretch",
            )
        with metric_col2:
            st.metric(
                "Target field",
                sections.get("Target Field", "Not mentioned"),
                border=True,
                height="stretch",
            )
        with gauge_col:
            with st.container(border=True, height="stretch"):
                st.echarts_chart(
                    build_score_gauge_spec(score),
                    height=180,
                    alt=f"Job fit score gauge showing {score} out of 100",
                )

        edu_col, exp_col = st.columns(2)
        with edu_col:
            with st.container(border=True, height="stretch"):
                st.markdown("**:material/school: Education**")
                st.write(sections.get("Education", "Not mentioned"))
        with exp_col:
            with st.container(border=True, height="stretch"):
                st.markdown("**:material/work: Experience**")
                st.write(sections.get("Experience", "Not mentioned"))

        with st.container(border=True):
            st.markdown("**:material/build: Skills**")
            skills_text = sections.get("Skills", "Not mentioned")
            skill_items = split_items(skills_text)
            if skill_items and len(skill_items) <= 20:
                with st.container(horizontal=True):
                    for skill in skill_items:
                        st.badge(skill, color="blue")
            else:
                st.write(skills_text)

        str_col, gap_col = st.columns(2)
        with str_col:
            st.success(
                sections.get("Strengths", "Not mentioned"),
                title="Strengths",
                icon=":material/thumb_up:",
            )
        with gap_col:
            st.warning(
                sections.get("Missing Skills", "Not mentioned"),
                title="Missing skills",
                icon=":material/trending_up:",
            )

        tab_report, tab_resume = st.tabs(
            ["Structured report", "Extracted resume text"],
            on_change="rerun",
        )
        with tab_report:
            st.code(raw_result, language="text")
        with tab_resume:
            st.text(resume_text)

        st.download_button(
            label="Download analysis report",
            data=raw_result,
            file_name=f"{Path(file_name).stem}_analysis.txt",
            mime="text/plain",
            icon=":material/download:",
            width="stretch",
            key="download_report_btn",
        )
else:
    st.info(
        "Upload a PDF resume above or select a sample resume in the sidebar to begin.",
        icon=":material/upload_file:",
    )

about_expander = st.expander("About ResumeAI", icon=":material/info:", on_change="rerun")
if about_expander.open:
    with about_expander:
        st.markdown(
            "ResumeAI extracts text from PDF resumes with **PyMuPDF** and runs a "
            "**Gemma 2B Instruct** model fine-tuned with **LoRA (PEFT)** to produce "
            "eight structured fields: Candidate Name, Target Field, Education, Experience, "
            "Skills, Strengths, Missing Skills, and Job Fit Score."
        )
