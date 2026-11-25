# 📄 Resume Analyser LLM

An AI-powered Resume Analysis application built using a fine-tuned Gemma 2B language model with LoRA.

The application allows users to upload a resume in PDF format and automatically analyzes the resume to provide structured information about the candidate.

---

## 🚀 Features

- 📄 PDF Resume Upload
- 🔍 Automatic Resume Text Extraction
- 🤖 AI-powered Resume Analysis
- 👤 Candidate Name Detection
- 🎯 Target Field Detection
- 🎓 Education Analysis
- 💼 Experience Analysis
- 🛠 Skills Analysis
- 💪 Strength Identification
- 📈 Missing Skills Detection
- 📊 Job Fit Score
- 📥 Downloadable Analysis
- 📱 Responsive Streamlit Interface
- ⚡ Local AI inference
- 🧠 Gemma 2B + LoRA fine-tuning

---

## 🏗️ Project Architecture

```text
Resume PDF
     │
     ▼
PDF Upload
     │
     ▼
PyMuPDF
     │
     ▼
Resume Text Extraction
     │
     ▼
Gemma 2B + LoRA
     │
     ▼
AI Resume Analysis
     │
     ├── Candidate Name
     ├── Target Field
     ├── Education
     ├── Experience
     ├── Skills
     ├── Strengths
     ├── Missing Skills
     └── Job Fit Score
     │
     ▼
Streamlit Dashboard
     │
     ▼
Download Analysis