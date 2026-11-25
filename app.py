from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st

from resume_parser import extract_text_from_pdf
from trained_model import (
    analyze_resume,
    extract_job_fit_score,
    get_hf_token,
    parse_analysis,
    INFERENCE_MODE,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResumeAI — Smart Resume Analyser",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (gradient hero, card polish, badge colours) ─────────────────────
st.html("""
<style>
/* ── Hero banner ─────────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 60%, #f093fb 100%);
    padding: 2.5rem 2rem;
    border-radius: 20px;
    color: #fff;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(102,126,234,.35);
}
.hero-banner h1 {
    font-size: 2.6rem;
    font-weight: 800;
    margin: 0 0 .4rem;
    letter-spacing: -.5px;
    text-shadow: 0 2px 8px rgba(0,0,0,.18);
}
.hero-banner p {
    font-size: 1.1rem;
    opacity: .92;
    margin: 0;
    max-width: 600px;
    margin-inline: auto;
}

/* ── Metric cards ────────────────────────────────────────── */
[data-testid="stMetric"] {
    border-radius: 14px;
    transition: transform .2s;
}
[data-testid="stMetric"]:hover { transform: translateY(-2px); }

/* ── Skill badge pill ────────────────────────────────────── */
.skill-pill {
    display: inline-block;
    background: linear-gradient(90deg,#667eea,#764ba2);
    color: #fff;
    border-radius: 999px;
    padding: .22rem .75rem;
    font-size: .82rem;
    font-weight: 600;
    margin: .18rem .18rem;
    box-shadow: 0 2px 6px rgba(102,126,234,.25);
}

/* ── Gauge container ─────────────────────────────────────── */
.gauge-wrap {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
}

/* ── Upload dropzone ─────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border-radius: 16px;
}

/* ── Download button ─────────────────────────────────────── */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(90deg,#667eea,#764ba2);
    color: #fff;
    border: none;
    border-radius: 10px;
    font-weight: 700;
}
[data-testid="stDownloadButton"] > button:hover {
    opacity: .88;
    transform: translateY(-1px);
}

/* ── Sidebar polish ──────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#1a1a2e 0%,#16213e 60%,#0f3460 100%);
    color: #e0e0ff;
}
[data-testid="stSidebar"] * { color: #e0e0ff !important; }
[data-testid="stSidebar"] .stMarkdown a { color: #a78bfa !important; }

/* ── Status widget ───────────────────────────────────────── */
[data-testid="stStatusWidget"] { border-radius: 12px; }

/* ── HF auth banner ──────────────────────────────────────── */
.hf-banner {
    background: linear-gradient(90deg,#ff9a44,#fc6076);
    color: #fff;
    padding: 1rem 1.4rem;
    border-radius: 14px;
    margin-bottom: 1.2rem;
    font-weight: 600;
    font-size: .95rem;
}
</style>
""")

# ── Paths ──────────────────────────────────────────────────────────────────────
SAMPLE_DIR = Path(__file__).resolve().parent / "sample_resumes"


# ── Helpers ────────────────────────────────────────────────────────────────────
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
                "title": {
                    "show": True,
                    "offsetCenter": [0, "65%"],
                    "fontSize": 13,
                    "color": "#888",
                },
                "detail": {
                    "valueAnimation": True,
                    "offsetCenter": [0, "5%"],
                    "fontSize": 28,
                    "fontWeight": "bold",
                    "formatter": "{value}/100",
                    "color": accent,
                },
                "data": [{"value": score, "name": "Job fit score"}],
            }
        ]
    }


def init_session_state() -> None:
    """Initialize session state defaults in one place."""
    st.session_state.setdefault("analysis_cache", {})
    st.session_state.setdefault("active_resume_key", None)


# ── Session init ───────────────────────────────────────────────────────────────
init_session_state()

# ── Hugging Face token check ───────────────────────────────────────────────────
hf_token: str | None = get_hf_token()

# ── Sidebar ────────────────────────────────────────────────────────────────────
sample_files = sorted(SAMPLE_DIR.glob("*.pdf")) if SAMPLE_DIR.is_dir() else []
sample_options: dict[str, Path | None] = {"None (upload your own PDF)": None}
for sample_path in sample_files:
    label = sample_path.stem.replace("_", " ").title()
    sample_options[label] = sample_path

with st.sidebar:
    st.markdown("## 📄 ResumeAI")
    st.caption("Structured resume analysis powered by **Gemma 2B + LoRA**.")

    st.divider()

    # HF Token status in sidebar
    if hf_token:
        mode_label = "🌐 API mode" if INFERENCE_MODE == "api" else "💻 Local mode"
        st.success(f"🔑 HF token loaded · {mode_label}", icon=":material/check_circle:")
    else:
        st.warning("⚠️ No HF token — set `HF_TOKEN` in secrets", icon=":material/key:")
        with st.expander("How to add your token"):
            st.markdown(
                "1. Get a token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)\n"
                "2. Add it to `.streamlit/secrets.toml`:\n"
                "```toml\nHF_TOKEN = \"hf_...\"\n```\n"
                "3. Or set env var: `export HF_TOKEN=hf_...`"
            )

    st.divider()

    selected_sample_label = st.selectbox(
        "Sample resume",
        options=list(sample_options.keys()),
        key="sample_resume_choice",
        help="Pick a bundled sample PDF or leave unselected to upload your own.",
    )

    st.divider()
    st.subheader("✨ Capabilities")
    st.markdown(
        "- Candidate & target role extraction\n"
        "- Education & experience summary\n"
        "- Skills & missing skills detection\n"
        "- Job fit score (0–100)\n"
        "- Downloadable plain-text report"
    )

    st.subheader("🧠 Model stack")
    st.caption("Gemma 2B Instruct · PEFT LoRA · PyMuPDF · Streamlit")

# ── Hero banner ────────────────────────────────────────────────────────────────
st.html("""
<div class="hero-banner">
  <h1>🎯 AI Resume Analyser</h1>
  <p>Upload a resume PDF and get an instant structured candidate assessment
     powered by a fine-tuned Gemma 2B + LoRA model.</p>
</div>
""")

# ── File upload ────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload resume PDF",
    type=["pdf"],
    key="resume_uploader",
    help="Select a text-based PDF resume (not scanned images).",
)

pdf_bytes: bytes | None = None
file_name: str | None = None

if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name
elif sample_options.get(selected_sample_label) is not None:
    sample_path = sample_options[selected_sample_label]
    if sample_path is not None and sample_path.is_file():
        pdf_bytes = sample_path.read_bytes()
        file_name = sample_path.name

# ── Main analysis flow ─────────────────────────────────────────────────────────
if pdf_bytes is not None and file_name is not None:
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()
    resume_key = f"{file_name}:{file_hash}"
    file_size_kb = len(pdf_bytes) / 1024

    with st.container(border=True):
        info_col, action_col = st.columns([3, 1], vertical_alignment="center")
        with info_col:
            st.markdown(f"**📎 {file_name}**")
            st.caption(f"Ready for analysis · {file_size_kb:.1f} KB")
        with action_col:
            analyze_clicked = st.button(
                "Analyse resume",
                key="analyze_btn",
                type="primary",
                icon=":material/analytics:",
                width="stretch",
            )

    if analyze_clicked:
        with st.status("🔍 Analysing resume...", expanded=True) as status:
            st.write("Extracting text from PDF…")
            resume_text = extract_resume_text_cached(pdf_bytes)

            if not resume_text.strip():
                status.update(label="Could not extract readable text", state="error")
                st.error(
                    "The PDF does not contain selectable text. "
                    "Scanned or image-only PDFs require OCR.",
                    icon=":material/error:",
                )
            else:
                try:
                    st.write("Loading Gemma 2B + LoRA model (first run may take ~1 min)…")
                    get_cached_model()

                    st.write("Generating structured analysis…")
                    raw_result = analyze_resume(resume_text)
                    sections = parse_analysis(raw_result)
                except Exception as exc:
                    exc_str = str(exc)
                    status.update(label="Model load failed", state="error")

                    # ── Gated / auth errors ───────────────────────────────
                    is_gated = any(k in exc_str for k in (
                        "GatedRepoError", "gated", "401", "403",
                        "Cannot access gated repo", "HTTPStatusError",
                        "token", "credential",
                    ))
                    if is_gated:
                        st.error(
                            "**🔒 Hugging Face authentication failed**\n\n"
                            "The model `google/gemma-2b-it` is **gated** — you need to:\n\n"
                            "1. ✅ [Accept the Gemma licence](https://huggingface.co/google/gemma-2b-it) "
                            "(one-time, free, click **Agree and access repository**)\n"
                            "2. 🔑 Get a **READ** token at "
                            "[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)\n"
                            "3. 📋 Add it to your deployment secrets:\n"
                            "   - **Streamlit Cloud:** App → ⋮ → Settings → Secrets → "
                            "`HF_TOKEN = \"hf_...\"`\n"
                            "   - **HF Spaces:** Settings → Variables and secrets → "
                            "`HF_TOKEN = \"hf_...\"`\n"
                            "4. 🔄 **Reboot the app** after saving the secret",
                            icon=":material/lock:",
                        )
                    else:
                        st.error(
                            f"**Model error:** {exc_str[:400]}\n\n"
                            "Check the app logs for the full traceback.",
                            icon=":material/error:",
                        )
                    st.stop()
                else:
                    pass  # fall through only on success

                st.session_state["analysis_cache"][resume_key] = {
                    "resume_text": resume_text,
                    "raw_result": raw_result,
                    "sections": sections,
                }
                st.session_state["active_resume_key"] = resume_key
                status.update(label="✅ Analysis complete!", state="complete")

    cached_entry = st.session_state["analysis_cache"].get(resume_key)
    if cached_entry is not None:
        resume_text = cached_entry["resume_text"]
        raw_result = cached_entry["raw_result"]
        sections = cached_entry["sections"]
        score = extract_job_fit_score(sections.get("Job Fit Score", "0"))

        st.subheader("📊 Assessment overview", divider="rainbow")

        # ── Top metrics row ──────────────────────────────────────────────────
        metric_col1, metric_col2, gauge_col = st.columns([1, 1, 1.2])
        with metric_col1:
            st.metric(
                "👤 Candidate",
                sections.get("Candidate Name", "Not mentioned"),
                border=True,
            )
        with metric_col2:
            st.metric(
                "🎯 Target field",
                sections.get("Target Field", "Not mentioned"),
                border=True,
            )
        with gauge_col:
            with st.container(border=True):
                try:
                    st.echarts_chart(
                        build_score_gauge_spec(score),
                        height=190,
                        alt=f"Job fit score gauge showing {score} out of 100",
                    )
                except AttributeError:
                    # Fallback for older Streamlit versions without echarts_chart
                    import plotly.graph_objects as go  # type: ignore

                    color = "#22C55E" if score >= 75 else ("#F59E0B" if score >= 50 else "#EF4444")
                    fig = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=score,
                            title={"text": "Job fit score"},
                            gauge={
                                "axis": {"range": [0, 100]},
                                "bar": {"color": color},
                            },
                        )
                    )
                    fig.update_layout(height=200, margin={"t": 40, "b": 0, "l": 20, "r": 20})
                    st.plotly_chart(fig, alt="Job fit score gauge")

        # ── Education / Experience ───────────────────────────────────────────
        edu_col, exp_col = st.columns(2)
        with edu_col:
            with st.container(border=True):
                st.markdown("**:material/school: Education**")
                st.write(sections.get("Education", "Not mentioned"))
        with exp_col:
            with st.container(border=True):
                st.markdown("**:material/work: Experience**")
                st.write(sections.get("Experience", "Not mentioned"))

        # ── Skills badges ────────────────────────────────────────────────────
        with st.container(border=True):
            st.markdown("**:material/build: Skills**")
            skills_text = sections.get("Skills", "Not mentioned")
            skill_items = split_items(skills_text)
            if skill_items and len(skill_items) <= 25:
                pills_html = "".join(
                    f'<span class="skill-pill">{s}</span>' for s in skill_items
                )
                st.html(f"<div>{pills_html}</div>")
            else:
                st.write(skills_text)

        # ── Strengths / Missing skills ───────────────────────────────────────
        str_col, gap_col = st.columns(2)
        with str_col:
            st.success(
                sections.get("Strengths", "Not mentioned"),
                icon=":material/thumb_up:",
            )
        with gap_col:
            st.warning(
                sections.get("Missing Skills", "Not mentioned"),
                icon=":material/trending_up:",
            )

        # ── Tabs ─────────────────────────────────────────────────────────────
        tab_report, tab_resume = st.tabs(["📋 Structured report", "📄 Extracted resume text"])
        with tab_report:
            st.code(raw_result, language="text")
        with tab_resume:
            st.text(resume_text)

        st.download_button(
            label="⬇️ Download analysis report",
            data=raw_result,
            file_name=f"{Path(file_name).stem}_analysis.txt",
            mime="text/plain",
            width="stretch",
            key="download_report_btn",
        )
else:
    st.info(
        "Upload a PDF resume above or select a sample resume in the sidebar to begin.",
        icon=":material/upload_file:",
    )

# ── About expander ─────────────────────────────────────────────────────────────
st.divider()
with st.expander("ℹ️ About ResumeAI"):
    st.markdown(
        "ResumeAI extracts text from PDF resumes with **PyMuPDF** and runs a "
        "**Gemma 2B Instruct** model fine-tuned with **LoRA (PEFT)** to produce "
        "eight structured fields: *Candidate Name, Target Field, Education, Experience, "
        "Skills, Strengths, Missing Skills,* and *Job Fit Score*.\n\n"
        "**Model adapter:** `riju12/resume-analyser-gemma-lora` (auto-falls back "
        "to local `gemma-resume-analyser/` if present).\n\n"
        "**HF authentication:** Required to download `google/gemma-2b-it`. "
        "Add your token to `.streamlit/secrets.toml` as `HF_TOKEN = \"hf_...\"` "
        "or export `HF_TOKEN` as an environment variable."
    )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.html("""
<div style="text-align:center;margin-top:2rem;padding:1.2rem;
            border-top:1px solid rgba(128,128,128,.2);
            color:#888;font-size:.82rem;">
  Built with ❤️ using Streamlit · Gemma 2B · PEFT LoRA · PyMuPDF<br>
  <a href="https://github.com/ranaghatanirban-pixel/resume-analyser-llm"
     style="color:#a78bfa;text-decoration:none;">
    ⭐ Star on GitHub
  </a>
</div>
""")
