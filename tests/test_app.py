from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from resume_parser import extract_text_from_pdf, remove_duplicate_lines, remove_repeated_text
from trained_model import (
    REQUIRED_SECTIONS,
    detect_section,
    extract_job_fit_score,
    format_analysis,
    parse_analysis,
)

SAMPLE_ANALYSIS = """Candidate Name:
Arupam Saha

Target Field:
AI / ML

Education:
B.Tech CSE (AI & ML), Adamas University

Experience:
Freelance Web & Automation Developer

Skills:
Python, SQL, Selenium, XGBoost

Strengths:
Strong Python automation and ML modeling background.

Missing Skills:
Docker, Kubernetes, Cloud Deployment

Job Fit Score:
88/100"""


class ResumeParserTests(unittest.TestCase):
    def test_remove_repeated_and_duplicate_lines(self) -> None:
        raw = "Line A\nLine A\nLine B\nline a\nLine C"
        no_repeat = remove_repeated_text(raw)
        self.assertEqual(no_repeat.splitlines(), ["Line A", "Line B", "line a", "Line C"])
        deduped = remove_duplicate_lines(no_repeat)
        self.assertEqual(deduped.splitlines(), ["Line A", "Line B", "Line C"])

    def test_extract_text_from_sample_pdf_path_and_bytes(self) -> None:
        pdf_path = Path("sample_resumes/resume_01_data_analyst.pdf")
        if not pdf_path.is_file():
            self.skipTest("Sample resume PDF not present")

        from_path = extract_text_from_pdf(pdf_path)
        from_bytes = extract_text_from_pdf(pdf_path.read_bytes())
        self.assertTrue(from_path.strip())
        self.assertEqual(from_path, from_bytes)


class ModelFormattingTests(unittest.TestCase):
    def test_missing_skills_not_split_by_skills(self) -> None:
        raw = (
            "**Candidate Name:** Jane Doe "
            "**Target Fields:** Data Science "
            "Education: B.Sc CS "
            "Experience: 2 years "
            "Skills: Python, SQL "
            "Strengths: Analytics "
            "Missing Skills: Docker, AWS "
            "Job fit score: 85/100"
        )
        formatted = format_analysis(raw)
        parsed = parse_analysis(formatted)

        for section in REQUIRED_SECTIONS:
            self.assertIn(section, parsed)

        self.assertEqual(parsed["Candidate Name"], "Jane Doe")
        self.assertEqual(parsed["Target Field"], "Data Science")
        self.assertEqual(parsed["Skills"], "Python, SQL")
        self.assertEqual(parsed["Missing Skills"], "Docker, AWS")
        self.assertEqual(parsed["Job Fit Score"], "85/100")

    def test_detect_section_handles_markdown_and_aliases(self) -> None:
        self.assertEqual(detect_section("**Target Fields:** ML"), "Target Field")
        self.assertEqual(detect_section("Job fit score: 90/100"), "Job Fit Score")
        self.assertEqual(detect_section("- Missing Skills: Kubernetes"), "Missing Skills")
        self.assertIsNone(detect_section("Unrelated text: hello"))

    def test_extract_job_fit_score_clamps_values(self) -> None:
        self.assertEqual(extract_job_fit_score("88/100"), 88)
        self.assertEqual(extract_job_fit_score("120%"), 100)
        self.assertEqual(extract_job_fit_score("Not mentioned"), 0)


class StreamlitAppTests(unittest.TestCase):
    def test_app_renders_and_analyses_sample_resume(self) -> None:
        app_path = str(Path(__file__).resolve().parent.parent / "app.py")
        with (
            patch("trained_model.load_model_and_tokenizer", return_value=(None, None, "cpu")),
            patch("trained_model.analyze_resume", return_value=SAMPLE_ANALYSIS),
        ):
            at = AppTest.from_file(app_path, default_timeout=10).run()
            self.assertFalse(at.exception)
            self.assertTrue(at.info)

            # Pick the first sample resume from the sidebar selectbox
            options = at.sidebar.selectbox(key="sample_resume_choice").options
            sample_options = [opt for opt in options if opt != "None (upload your own PDF)"]
            if not sample_options:
                self.skipTest("No sample resumes available")

            at.sidebar.selectbox(key="sample_resume_choice").select(sample_options[0]).run()
            self.assertFalse(at.exception)

            at.button(key="analyze_btn").click().run()
            self.assertFalse(at.exception)
            self.assertEqual(at.metric[0].value, "Arupam Saha")
            self.assertEqual(at.metric[1].value, "AI / ML")
            self.assertTrue(at.success)
            self.assertTrue(at.warning)
            self.assertTrue(at.get("echarts_chart"))


if __name__ == "__main__":
    unittest.main()
