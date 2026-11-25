from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_MODEL = os.getenv("BASE_MODEL", "google/gemma-2b-it")
DEFAULT_HF_ADAPTER = "riju12/resume-analyser-gemma-lora"
LOCAL_ADAPTER_DIR = Path(__file__).resolve().parent / "gemma-resume-analyser"

# Default to "local" for local Mac execution; set INFERENCE_MODE=api for HF API.
INFERENCE_MODE = os.getenv("INFERENCE_MODE", "local").lower()

REQUIRED_SECTIONS = [
    "Candidate Name",
    "Target Field",
    "Education",
    "Experience",
    "Skills",
    "Strengths",
    "Missing Skills",
    "Job Fit Score",
]

SECTION_ALIASES = {
    "candidate name": "Candidate Name",
    "candidate": "Candidate Name",
    "name": "Candidate Name",
    "target field": "Target Field",
    "target fields": "Target Field",
    "target role": "Target Field",
    "education": "Education",
    "experience": "Experience",
    "work experience": "Experience",
    "skills": "Skills",
    "technical skills": "Skills",
    "strengths": "Strengths",
    "missing skills": "Missing Skills",
    "job fit score": "Job Fit Score",
    "job-fit score": "Job Fit Score",
    "fit score": "Job Fit Score",
}

_SORTED_ALIASES = sorted(SECTION_ALIASES.keys(), key=len, reverse=True)
_alias_group = "(" + "|".join(re.escape(a) for a in _SORTED_ALIASES) + ")"
# Use [*]{0,2} instead of \*{0,2} — the latter is rejected by Python 3.14+
_HEADING_PATTERN = re.compile(
    r"(?im)^\s*[*]{0,2}" + _alias_group + r"[*]{0,2}\s*:\s*[*]{0,2}(.*)$"
)
_INLINE_HEADING_PATTERN = re.compile(
    r"(?i)(?<!\w)" + _alias_group + r"\s*:"
)

_MODEL_CACHE: tuple[Any, Any, str] | None = None


# ── Token helper ───────────────────────────────────────────────────────────────
def get_hf_token() -> str | None:
    """Return a Hugging Face token from env, Streamlit secrets, or hf cache."""
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    try:
        import streamlit as st  # noqa: PLC0415
        secret_token = st.secrets.get("HF_TOKEN")
        if secret_token:
            return str(secret_token)
    except Exception:
        pass
    try:
        from huggingface_hub import HfFolder  # noqa: PLC0415
        cached = HfFolder.get_token()
        if cached:
            return cached
    except Exception:
        pass
    return None


# ── Prompt builder ─────────────────────────────────────────────────────────────
def build_prompt_text(resume_text: str) -> str:
    """Build the structured analysis prompt (plain text, no tokenizer needed)."""
    return (
        "Analyze the resume below and return a structured professional assessment.\n"
        "Use EXACTLY these 8 headings, each once, in this order:\n"
        "Candidate Name:\n"
        "Target Field:\n"
        "Education:\n"
        "Experience:\n"
        "Skills:\n"
        "Strengths:\n"
        "Missing Skills:\n"
        "Job Fit Score:\n\n"
        "Rules:\n"
        "- Write the value on the same line or directly below each heading.\n"
        '- If information is unavailable, write "Not mentioned".\n'
        "- Job Fit Score must be an integer out of 100, e.g. 85/100.\n"
        "- Keep each section concise and in plain text (no markdown).\n\n"
        f"Resume:\n{resume_text}\n\nAnswer:\n"
    )


# ── HF Inference API mode (default for cloud) ──────────────────────────────────
def _analyze_via_api(resume_text: str, hf_token: str) -> str:
    """Call the HuggingFace Inference API — no local GPU or large RAM needed."""
    from huggingface_hub import InferenceClient  # noqa: PLC0415

    client = InferenceClient(token=hf_token)
    prompt = build_prompt_text(resume_text)

    try:
        response = client.text_generation(
            prompt,
            model=DEFAULT_HF_ADAPTER,
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.05,
            return_full_text=False,
        )
    except Exception:
        response = client.text_generation(
            prompt,
            model=BASE_MODEL,
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.05,
            return_full_text=False,
        )

    return format_analysis(response)


# ── Local model mode (for local dev with GPU or large RAM) ─────────────────────
def detect_device() -> str:
    import torch  # noqa: PLC0415
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_from_hf_with_fallback(loader: Any, model_id: str, **kwargs: Any) -> Any:
    offline_kwargs = dict(kwargs, local_files_only=True)
    try:
        return loader.from_pretrained(model_id, **offline_kwargs)
    except OSError:
        return loader.from_pretrained(model_id, **kwargs)


def load_model_and_tokenizer() -> tuple[Any, Any, str]:
    """Load and cache the Gemma model + LoRA adapter locally."""
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    import torch  # noqa: PLC0415
    from peft import PeftModel  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    device = detect_device()
    print(f"[Gemma] Loading on device: {device}", flush=True)

    adapter_path = (
        str(LOCAL_ADAPTER_DIR)
        if (LOCAL_ADAPTER_DIR / "adapter_config.json").is_file()
        else DEFAULT_HF_ADAPTER
    )
    hf_token = get_hf_token()
    if not hf_token:
        raise EnvironmentError(
            "GatedRepoError: HF_TOKEN is required for local model loading. "
            "Set HF_TOKEN in your secrets or use INFERENCE_MODE=api."
        )
    auth = {"token": hf_token}

    try:
        print("[Gemma] Loading tokenizer...", flush=True)
        tokenizer = _load_from_hf_with_fallback(AutoTokenizer, BASE_MODEL, **auth)
    except Exception as exc:
        raise EnvironmentError(f"GatedRepoError: {exc}") from exc

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if device in ("mps", "cuda") else torch.float32
    model_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        **auth,
    }
    if device == "cuda":
        model_kwargs["device_map"] = "auto"
    elif device == "mps":
        model_kwargs["device_map"] = {"": "mps"}

    try:
        print("[Gemma] Loading base model weights...", flush=True)
        base_model = _load_from_hf_with_fallback(AutoModelForCausalLM, BASE_MODEL, **model_kwargs)
    except Exception as exc:
        raise EnvironmentError(f"GatedRepoError: {exc}") from exc

    print(f"[Gemma] Attaching LoRA adapter from {adapter_path}...", flush=True)
    is_local = Path(adapter_path).exists()
    model = PeftModel.from_pretrained(base_model, adapter_path, **({} if is_local else auth))
    if device == "cpu":
        model = model.to(device)
    model.eval()

    print("[Gemma] Model ready!", flush=True)
    _MODEL_CACHE = (model, tokenizer, device)
    return _MODEL_CACHE


def _analyze_local(resume_text: str) -> str:
    """Run the fine-tuned Gemma + LoRA model locally."""
    import torch  # noqa: PLC0415

    model, tokenizer, device = load_model_and_tokenizer()

    user_content = build_prompt_text(resume_text)
    messages = [{"role": "user", "content": user_content}]
    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = user_content

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    target_device = model.device if device == "cuda" else device
    inputs = {k: v.to(target_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_length = inputs["input_ids"].shape[1]
    raw_output = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    return format_analysis(raw_output)


# ── Public entry-point ─────────────────────────────────────────────────────────
def analyze_resume(resume_text: str) -> str:
    """
    Analyse a resume. Chooses mode:
    - INFERENCE_MODE=local → load model locally (default)
    - INFERENCE_MODE=api   → HF Inference API
    """
    hf_token = get_hf_token()
    if not hf_token:
        raise EnvironmentError(
            "GatedRepoError: No HF_TOKEN found. "
            "Set HF_TOKEN in your Streamlit Cloud secrets (or as an env var). "
            "Get a token at https://huggingface.co/settings/tokens and accept the "
            "Gemma licence at https://huggingface.co/google/gemma-2b-it"
        )

    if INFERENCE_MODE == "local":
        return _analyze_local(resume_text)
    return _analyze_via_api(resume_text, hf_token)


# ── Text normalization helpers ─────────────────────────────────────────────────
def format_analysis(result: str) -> str:
    """Normalize model output into 8 canonical sections without duplicates."""
    normalized = result.replace("\r\n", "\n").replace("\r", "\n").strip()

    lines_prepared: list[str] = []
    for raw_line in normalized.split("\n"):
        cleaned_line = raw_line.strip()
        if not cleaned_line:
            continue
        split_line = _INLINE_HEADING_PATTERN.sub(
            lambda m: f"\n{m.group(1)}:", cleaned_line
        )
        for sub_line in split_line.split("\n"):
            if sub_line.strip():
                lines_prepared.append(sub_line.strip())

    section_data: dict[str, list[str]] = {s: [] for s in REQUIRED_SECTIONS}
    current_section: str | None = None
    seen_sections: set[str] = set()

    for line in lines_prepared:
        match = _HEADING_PATTERN.match(line)
        if match:
            canonical = SECTION_ALIASES.get(match.group(1).strip().lower())
            if canonical is None:
                continue
            if canonical in seen_sections:
                current_section = None
                continue
            seen_sections.add(canonical)
            current_section = canonical
            inline_content = match.group(2).strip().strip("*")
            if inline_content:
                section_data[canonical].append(inline_content)
            continue
        if current_section is not None:
            cleaned_item = re.sub(r"^[-*•]\s+", "", line).strip()
            if cleaned_item:
                section_data[current_section].append(cleaned_item)

    final_output: list[str] = []
    for section in REQUIRED_SECTIONS:
        seen_items: set[str] = set()
        unique_items: list[str] = []
        for item in section_data[section]:
            key = item.lower()
            if key not in seen_items:
                seen_items.add(key)
                unique_items.append(item)
        final_output.append(f"{section}:")
        final_output.append(" ".join(unique_items) if unique_items else "Not mentioned")
        final_output.append("")

    return "\n".join(final_output).strip()


def parse_analysis(result: str) -> dict[str, str]:
    """Parse formatted analysis text into a dict keyed by section name."""
    sections: dict[str, str] = {s: "" for s in REQUIRED_SECTIONS}
    current_section: str | None = None

    for raw_line in result.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _HEADING_PATTERN.match(line)
        if match:
            canonical = SECTION_ALIASES.get(match.group(1).strip().lower())
            if canonical is not None:
                current_section = canonical
                content = match.group(2).strip()
                if content:
                    sections[current_section] = content
                continue
        if current_section is not None:
            if sections[current_section]:
                sections[current_section] += " " + line
            else:
                sections[current_section] = line

    for section in REQUIRED_SECTIONS:
        if not sections[section].strip():
            sections[section] = "Not mentioned"
    return sections


def extract_job_fit_score(score_text: str) -> int:
    """Extract an integer job-fit score clamped to [0, 100]."""
    cleaned = score_text.replace("%", "").replace("/100", "").replace("out of 100", "").strip()
    match = re.search(r"\d+", cleaned)
    if not match:
        return 0
    return max(0, min(int(match.group()), 100))