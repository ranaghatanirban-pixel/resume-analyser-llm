# Resume Analyser LLM

## 1. Project Overview

Resume Analyser LLM is an AI-powered resume analysis application built with Streamlit that extracts text from PDF resumes and generates a structured candidate assessment using a fine-tuned **Google Gemma 2B Instruct** model with **LoRA (PEFT)**.

---

## 2. Features

- PDF resume upload or bundled sample resume selection
- Automatic PDF text extraction and deduplication via PyMuPDF
- Structured extraction across 8 sections:
  1. Candidate Name
  2. Target Field
  3. Education
  4. Experience
  5. Skills
  6. Strengths
  7. Missing Skills
  8. Job Fit Score (0–100 gauge visualization)
- Session-cached analysis results and downloadable plain-text report
- Automatic local-first adapter resolution with Hugging Face Hub fallback

---

## 3. System Architecture

```text
Resume PDF
      ↓
Streamlit Application (app.py)
      ↓
PyMuPDF PDF Parser (resume_parser.py)
      ↓
Extracted Resume Text
      ↓
Fine-Tuned Gemma 2B + LoRA (trained_model.py)
      ↓
Output Normalization & Section Parsing
      ↓
Interactive Dashboard & Downloadable Report
```

---

## 4. Project Structure

```text
AI resume analyser/
├── .streamlit/
│   └── config.toml
├── app.py
├── trained_model.py
├── resume_parser.py
├── llm_analyzer.py
├── test_pipeline.py
├── check_dataset.py
├── requirements.txt
├── README.md
├── dataset/
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── test.jsonl
├── sample_resumes/
├── tests/
│   └── test_app.py
├── evaluation_results.jsonl
├── evaluation_summary.json
└── gemma-resume-analyser/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── tokenizer files
```

---

## 5. Installation & Setup

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

If `google/gemma-2b-it` is not already cached locally, provide your Hugging Face token via environment variable or `.streamlit/secrets.toml` (never commit `secrets.toml`):

```bash
export HF_TOKEN="your_huggingface_token"
```

---

## 6. Run the Application

Start Streamlit:

```bash
streamlit run app.py
```

---

## 7. Testing

Run the fast unit & Streamlit `AppTest` suite:

```bash
python3 -m unittest discover -s tests -v
```

Run the full end-to-end model pipeline test on a PDF:

```bash
python3 test_pipeline.py resume.pdf
```

---

## 8. Model & Evaluation

- **Base model:** `google/gemma-2b-it`
- **Fine-tuning method:** LoRA (`peft`)
- **Adapter location:** `gemma-resume-analyser/` (automatically preferred locally, falling back to `riju12/resume-analyser-gemma-lora`)
- **Evaluation summary (74 test cases):**
  - Cases with all sections: 68/74 (`91.89%`)
  - Corrupted term cases: 0/74
  - ROUGE-1: `0.7362` | ROUGE-L: `0.6639`
