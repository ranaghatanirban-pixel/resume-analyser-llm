# ============================================================
# ResumeAI — AI Resume Analyser
# README
# ============================================================

<div align="center">

# 🎯 ResumeAI — AI Resume Analyser

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![HuggingFace](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

**An AI-powered resume analyser built with Streamlit + fine-tuned Gemma 2B + LoRA**

[🚀 Live Demo](#deployment) · [📖 Docs](#setup) · [🐛 Issues](https://github.com/ranaghatanirban-pixel/resume-analyser-llm/issues)

</div>

---

## ✨ Features

| Feature | Detail |
|---|---|
| 📄 PDF parsing | PyMuPDF — fast, robust text extraction + deduplication |
| 🧠 AI analysis | Gemma 2B Instruct fine-tuned with LoRA (PEFT) |
| 📊 8 structured fields | Name, Field, Education, Experience, Skills, Strengths, Missing Skills, Job Fit Score |
| 🎯 Score gauge | Interactive ECharts gauge (0–100) with colour coding |
| 🔖 Skill badges | Gradient pill badges for each detected skill |
| ⬇️ Download report | Plain-text report downloadable in one click |
| 🌙 Dark/Light mode | Auto-adapts via Streamlit config |
| 🔑 HF Auth | Token loaded from secrets or env — zero friction for users |
| 📦 Sample resumes | 5 bundled sample PDFs for instant demo |

---

## 🏗️ Architecture

```
Resume PDF
      ↓
Streamlit App (app.py)
      ↓
PyMuPDF PDF Parser (resume_parser.py)
      ↓
Extracted Resume Text
      ↓
Fine-Tuned Gemma 2B + LoRA (trained_model.py)
      ↓
Output Normalization & Section Parsing
      ↓
Interactive Dashboard + Downloadable Report
```

---

## 🚀 Quick Start

### 1. Clone

```bash
git clone https://github.com/ranaghatanirban-pixel/resume-analyser-llm.git
cd resume-analyser-llm
```

### 2. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Hugging Face token

> [!IMPORTANT]
> `google/gemma-2b-it` is a **gated model**. You must:
> 1. Create a free account at [huggingface.co](https://huggingface.co)
> 2. Accept the Gemma licence at [huggingface.co/google/gemma-2b-it](https://huggingface.co/google/gemma-2b-it)
> 3. Create a **READ** token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

Copy the example secrets file and fill in your token:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Then edit .streamlit/secrets.toml and replace hf_YOUR_TOKEN_HERE
```

Or export as environment variable:

```bash
export HF_TOKEN="hf_..."
```

### 5. Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🌐 Deployment

### Streamlit Community Cloud (Recommended — Free)

1. Push this repo to GitHub (already done ✅)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select this repo + `app.py`
4. In **Advanced settings → Secrets**, add:
   ```toml
   HF_TOKEN = "hf_..."
   ```
5. Click **Deploy** 🎉

### Hugging Face Spaces

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/new-space)
2. Choose **Streamlit** SDK
3. Push this repo (or link your GitHub repo)
4. Add `HF_TOKEN` in **Settings → Variables and secrets**

### Vercel

> [!NOTE]
> Streamlit is a persistent server; Vercel serverless is best for the frontend proxy.
> The included `vercel.json` and `api/index.py` provide a lightweight bridge.

```bash
npm i -g vercel
vercel --prod
```

Set `HF_TOKEN` in **Vercel Dashboard → Settings → Environment Variables**.

### Docker

```bash
docker build -t resumeai .
docker run -p 8501:8501 -e HF_TOKEN=hf_... resumeai
```

---

## 📁 Project Structure

```
resume-analyser-llm/
├── .streamlit/
│   ├── config.toml              # Theme & server config
│   └── secrets.toml.example     # Template (never commit secrets.toml)
├── api/
│   └── index.py                 # Vercel serverless bridge
├── assets/
│   └── style.css                # Legacy CSS (kept for reference)
├── sample_resumes/              # 5 bundled PDF samples
├── gemma-resume-analyser/       # Local LoRA adapter weights
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files
├── dataset/                     # Training data
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── test.jsonl
├── tests/
│   └── test_app.py
├── app.py                       # 🌟 Main Streamlit app
├── trained_model.py             # Model loading + inference
├── resume_parser.py             # PDF text extraction
├── llm_analyzer.py              # LLM utility helpers
├── requirements.txt
├── vercel.json                  # Vercel deployment config
└── README.md
```

---

## 🧠 Model Details

| Attribute | Value |
|---|---|
| Base model | `google/gemma-2b-it` |
| Fine-tuning | LoRA (PEFT) |
| Adapter (HF Hub) | `riju12/resume-analyser-gemma-lora` |
| Local adapter | `gemma-resume-analyser/` (auto-preferred) |
| Evaluation (74 cases) | 91.89% complete sections · ROUGE-1: 0.74 · ROUGE-L: 0.66 |

---

## 🔧 Configuration

### Environment variables

| Variable | Description | Required |
|---|---|---|
| `HF_TOKEN` | Hugging Face access token | ✅ Yes |
| `BASE_MODEL` | Override base model ID | ❌ No |
| `ADAPTER_PATH` | Override adapter path | ❌ No |
| `STREAMLIT_PORT` | Port for Vercel bridge | ❌ No |

### Streamlit secrets (`.streamlit/secrets.toml`)

```toml
HF_TOKEN = "hf_..."
```

---

## 🧪 Testing

```bash
# Fast unit + AppTest suite
python -m unittest discover -s tests -v

# End-to-end pipeline on a real PDF
python test_pipeline.py resume.pdf
```

---

## 📄 License

MIT © [ranaghatanirban-pixel](https://github.com/ranaghatanirban-pixel)

---

<div align="center">
Built with ❤️ using Streamlit · Gemma 2B · PEFT LoRA · PyMuPDF
</div>
