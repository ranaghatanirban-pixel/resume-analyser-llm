import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ============================================================
# MODEL CONFIGURATION
# ============================================================

BASE_MODEL = "google/gemma-2b-it"

ADAPTER_PATH = "./gemma-resume-analyser"


# ============================================================
# SELECT DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print("Using device:", DEVICE)


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# LOAD BASE GEMMA MODEL
# ============================================================

print("Loading base Gemma model...")

if DEVICE == "mps":
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16
    )

elif DEVICE == "cuda":
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )

else:
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32
    )


# ============================================================
# LOAD LORA ADAPTER
# ============================================================

print("Loading LoRA adapter...")

model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH
)


# ============================================================
# MOVE MODEL TO DEVICE
# ============================================================

if DEVICE == "mps":
    model = model.to("mps")

elif DEVICE == "cpu":
    model = model.to("cpu")


model.eval()

print("Model loaded successfully!")


# ============================================================
# REQUIRED SECTIONS
# ============================================================

REQUIRED_SECTIONS = [
    "Candidate Name:",
    "Target Field:",
    "Education:",
    "Experience:",
    "Skills:",
    "Strengths:",
    "Missing Skills:",
    "Job Fit Score:"
]


# ============================================================
# CLEAN MODEL OUTPUT
# ============================================================

def clean_analysis(result):

    result = result.strip()

    # --------------------------------------------------------
    # Remove repeated sections
    # Keep the first occurrence of each section.
    # --------------------------------------------------------

    lines = result.splitlines()

    cleaned_lines = []
    seen_sections = set()

    for line in lines:

        stripped = line.strip()

        matched_section = None

        for section in REQUIRED_SECTIONS:

            if stripped.lower().startswith(
                section.lower()
            ):
                matched_section = section
                break

        if matched_section:

            section_key = matched_section.lower()

            if section_key in seen_sections:
                continue

            seen_sections.add(section_key)

        cleaned_lines.append(line)

    result = "\n".join(cleaned_lines)

    return result.strip()


# ============================================================
# ANALYZE RESUME
# ============================================================

def analyze_resume(resume_text):

    prompt = f"""Instruction:
Analyze the following resume and return a structured professional assessment. Use exactly these sections: Candidate Name, Target Field, Education, Experience, Skills, Strengths, Missing Skills, and Job Fit Score. Do not repeat sections. Do not invent information that is not supported by the resume.

Resume:
{resume_text}

Expected Analysis:
"""

    # --------------------------------------------------------
    # Tokenize
    # --------------------------------------------------------

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )

    # --------------------------------------------------------
    # Move inputs to correct device
    # --------------------------------------------------------

    if DEVICE == "mps":
        inputs = {
            key: value.to("mps")
            for key, value in inputs.items()
        }

    elif DEVICE == "cuda":
        inputs = {
            key: value.to(model.device)
            for key, value in inputs.items()
        }

    else:
        inputs = {
            key: value.to("cpu")
            for key, value in inputs.items()
        }

    # --------------------------------------------------------
    # Generate response
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # --------------------------------------------------------
    # Remove prompt from generated output
    # --------------------------------------------------------

    input_length = inputs["input_ids"].shape[1]

    generated_tokens = outputs[0][input_length:]

    result = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    # --------------------------------------------------------
    # Clean repeated sections
    # --------------------------------------------------------

    result = clean_analysis(result)

    return result