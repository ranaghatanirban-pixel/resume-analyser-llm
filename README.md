# Resume Analyser LLM

## 1. Project Overview

Resume Analyser LLM is an AI-powered resume analysis application that analyzes PDF resumes and provides a structured professional assessment.

The application uses:

- Python
- Streamlit
- PyMuPDF
- Google Gemma 2B
- LoRA fine-tuning
- Hugging Face Transformers
- PyTorch

The system extracts text from an uploaded resume PDF and passes the extracted content to a fine-tuned Gemma model. The generated response is then formatted into a structured resume analysis.

---

## 2. Features

The application provides:

- PDF resume upload
- Automatic resume text extraction
- Candidate name identification
- Target field identification
- Education analysis
- Experience analysis
- Skills identification
- Strength identification
- Missing skills identification
- Job fit score
- Downloadable analysis
- Duplicate heading removal
- Handling of missing information

---

## 3. Output Format

The application produces the following sections:

1. Candidate Name
2. Target Field
3. Education
4. Experience
5. Skills
6. Strengths
7. Missing Skills
8. Job Fit Score

---

## 4. System Architecture

The project follows this pipeline:

Resume PDF
      ↓
Streamlit Application
      ↓
PyMuPDF PDF Parser
      ↓
Extracted Resume Text
      ↓
Fine-Tuned Gemma 2B + LoRA
      ↓
Generated Resume Analysis
      ↓
Output Formatting
      ↓
Duplicate Section Removal
      ↓
Final Resume Analysis

---

## 5. Project Structure

```text
resume-analyser/
│
├── app.py
├── trained_model.py
├── resume_parser.py
├── llm_analyzer.py
├── test_pipeline.py
├── requirements.txt
├── README.md
│
├── train_fixed.jsonl
├── validation_fixed.jsonl
├── test_fixed.jsonl
│
├── evaluation_results.jsonl
├── evaluation_summary.json
│
└── gemma-resume-analyser/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── tokenizer files
```

---

## 6. Model

Base model:

Google Gemma 2B Instruct

Fine-tuning method:

LoRA (Low-Rank Adaptation)

The model was fine-tuned using structured resume-analysis training data.

---

## 7. Dataset

The project uses separate datasets for:

- Training
- Validation
- Testing

The datasets contain resume text, instructions, and expected structured analysis.

Dataset format:

```json
{
  "instruction": "Analyze the following resume and return a structured professional assessment.",
  "input": "Resume text",
  "output": "Expected structured analysis"
}
```

---

## 8. Training

The Gemma model was fine-tuned using LoRA.

Training included:

- 4-bit quantization
- LoRA adapters
- Training dataset
- Validation dataset
- 3 training epochs

The trained adapter was saved in:

```text
gemma-resume-analyser/
```

---

## 9. Evaluation

The trained model was evaluated using 74 test cases.

Important evaluation results:

- Test cases: 74
- Required sections present: 68/74
- Section completeness: 91.89%
- Known corrupted/mixed-language terms: 0/74
- ROUGE-1: approximately 0.7362

The application also includes an output-cleaning layer to control duplicate headings and maintain a consistent final format.

---

## 10. Installation

Clone or copy the project into a local folder.

Open the project in VS Code.

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

---

## 11. Run the Application

Start Streamlit:

```bash
python3 -m streamlit run app.py
```

The application will open in the browser.

---

## 12. How to Use

1. Open the Resume Analyser application.
2. Upload a resume PDF.
3. Click "Analyze Resume".
4. Wait for the AI analysis.
5. Review the structured analysis.
6. Download the analysis if required.

---

## 13. Example Output

```text
Candidate Name:
John Smith

Target Field:
Data Analyst

Education:
B.Tech in Computer Science

Experience:
2 years

Skills:
Python, SQL, Excel, Power BI

Strengths:
Strong analytical and problem-solving skills

Missing Skills:
Advanced Machine Learning

Job Fit Score:
85/100
```

---

## 14. Limitations

The current system has some limitations:

- Image-only/scanned PDFs may not produce usable text because the current parser does not include OCR.
- Very long resumes may require additional text processing.
- AI-generated results should be reviewed by the user.
- Job fit scores are model-generated estimates and should not be treated as definitive hiring decisions.

---

## 15. Future Enhancements

Possible future improvements include:

- OCR support for scanned resumes
- Job description matching
- Resume ranking
- ATS score
- Skill recommendations
- Multiple resume comparison
- Database storage
- Cloud deployment
- Better visualization of job-fit results
- Improved model evaluation

---

## 16. Technologies Used

- Python
- Streamlit
- PyTorch
- Hugging Face Transformers
- PEFT
- LoRA
- Gemma 2B
- PyMuPDF
- JSONL datasets