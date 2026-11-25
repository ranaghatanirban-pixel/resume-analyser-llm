from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = os.getenv("BASE_MODEL", "google/gemma-2b-it")
DEFAULT_HF_ADAPTER = "riju12/resume-analyser-gemma-lora"
LOCAL_ADAPTER_DIR = Path(__file__).resolve().parent / "gemma-resume-analyser"

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

# Match longer aliases before shorter ones so "Missing Skills" is never split at "Skills".
_SORTED_ALIASES = sorted(SECTION_ALIASES.keys(), key=len, reverse=True)
_HEADING_PATTERN = re.compile(
    r"(?im)^(?:\s*(?:\d+[\.\)]\s*|[-*•]\s*|#+\s*)?)\*{0,2}("
    + "|".join(re.escape(alias) for alias in _SORTED_ALIASES)
    + r")\*{0,2}\s*:\s*\*{0,2}(.*)$"
)
_INLINE_HEADING_PATTERN = re.compile(
    r"(?i)(?<!\w)("
    + "|".join(re.escape(alias) for alias in _SORTED_ALIASES)
    + r")\s*:"
)

_MODEL_CACHE: tuple[Any, Any, str] | None = None


def resolve_adapter_path() -> str:
    """Resolve the LoRA adapter path, preferring local files when available."""
    env_adapter = os.getenv("ADAPTER_PATH")
    if env_adapter:
        return env_adapter
    if (LOCAL_ADAPTER_DIR / "adapter_config.json").is_file():
        return str(LOCAL_ADAPTER_DIR)
    return DEFAULT_HF_ADAPTER


def get_hf_token() -> str | None:
    """Return a Hugging Face token from environment variables or Streamlit secrets."""
    token = os.getenv("HF_TOKEN")
    if token:
        return token

    try:
        import streamlit as st

        secret_token = st.secrets.get("HF_TOKEN")
        if secret_token:
            return str(secret_token)
    except Exception:
        pass

    return None


def detect_device() -> str:
    """Detect the best available PyTorch compute device."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_from_hf_with_fallback(loader: Any, model_id: str, **kwargs: Any) -> Any:
    """Prefer locally cached Hugging Face artifacts before hitting the network."""
    offline_kwargs = dict(kwargs)
    offline_kwargs["local_files_only"] = True
    try:
        return loader.from_pretrained(model_id, **offline_kwargs)
    except OSError:
        return loader.from_pretrained(model_id, **kwargs)


def load_model_and_tokenizer() -> tuple[Any, Any, str]:
    """Load and cache the base Gemma model, tokenizer, and LoRA adapter."""
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    device = detect_device()
    adapter_path = resolve_adapter_path()
    hf_token = get_hf_token()
    auth_kwargs: dict[str, Any] = {"token": hf_token} if hf_token else {}

    tokenizer = _load_from_hf_with_fallback(
        AutoTokenizer,
        BASE_MODEL,
        **auth_kwargs,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
    model_kwargs: dict[str, Any] = {"dtype": dtype, **auth_kwargs}
    if device == "cuda":
        model_kwargs["device_map"] = "auto"

    base_model = _load_from_hf_with_fallback(
        AutoModelForCausalLM,
        BASE_MODEL,
        **model_kwargs,
    )

    is_local_adapter = Path(adapter_path).exists()
    adapter_kwargs = {} if is_local_adapter else dict(auth_kwargs)
    model = PeftModel.from_pretrained(base_model, adapter_path, **adapter_kwargs)

    if device in ("mps", "cpu"):
        model = model.to(device)

    model.eval()
    _MODEL_CACHE = (model, tokenizer, device)
    return _MODEL_CACHE


def detect_section(line: str) -> str | None:
    """Detect a canonical section heading from a single line of model output."""
    match = _HEADING_PATTERN.match(line.strip())
    if not match:
        return None
    alias = match.group(1).strip().lower()
    return SECTION_ALIASES.get(alias)


def format_analysis(result: str) -> str:
    """Normalize model output into the 8 canonical sections without duplicates."""
    normalized = result.replace("\r\n", "\n").replace("\r", "\n").strip()

    # Ensure inline headings like "... Skills: Python Missing Skills: AWS" start on new lines
    # without splitting inside multi-word headings such as "Missing Skills:".
    lines_prepared: list[str] = []
    for raw_line in normalized.split("\n"):
        cleaned_line = raw_line.strip()
        if not cleaned_line:
            continue
        split_line = _INLINE_HEADING_PATTERN.sub(
            lambda m: f"\n{m.group(1)}:",
            cleaned_line,
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
        unique_items: list[str] = []
        seen_items: set[str] = set()
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
    """Parse formatted analysis text into a dictionary keyed by section name."""
    sections: dict[str, str] = {section: "" for section in REQUIRED_SECTIONS}
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
    """Extract an integer job fit score clamped to [0, 100]."""
    cleaned = (
        score_text.replace("%", "")
        .replace("/100", "")
        .replace("out of 100", "")
        .strip()
    )
    match = re.search(r"\d+", cleaned)
    if not match:
        return 0
    return max(0, min(int(match.group()), 100))


def build_prompt(resume_text: str, tokenizer: Any) -> str:
    """Build the Gemma chat prompt for structured resume analysis."""
    user_content = (
        "Analyze the resume and return a structured professional assessment.\n"
        "Use EXACTLY these 8 headings, each once, in this exact order:\n"
        "Candidate Name:\n"
        "Target Field:\n"
        "Education:\n"
        "Experience:\n"
        "Skills:\n"
        "Strengths:\n"
        "Missing Skills:\n"
        "Job Fit Score:\n\n"
        "Rules:\n"
        "- Put the value on the same line or directly below each heading.\n"
        '- Do not invent information. If unavailable, write "Not mentioned".\n'
        "- Job Fit Score must be an integer out of 100 such as 85/100.\n"
        "- Keep each section concise and plain text (no markdown).\n\n"
        f"Resume:\n{resume_text}"
    )
    messages = [{"role": "user", "content": user_content}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"{user_content}\n\nAnswer:\n"


def analyze_resume(resume_text: str) -> str:
    """Run the fine-tuned Gemma + LoRA model on extracted resume text."""
    model, tokenizer, device = load_model_and_tokenizer()
    prompt = build_prompt(resume_text, tokenizer)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )
    target_device = model.device if device == "cuda" else device
    inputs = {key: value.to(target_device) for key, value in inputs.items()}

    # Avoid no_repeat_ngram_size here because it blocks copying exact names,
    # universities, and dates that appear in the input resume.
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
    generated_tokens = outputs[0][input_length:]
    raw_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return format_analysis(raw_output)
