# AI Resume Screening Tool

AI-powered resume screening application that ranks uploaded resumes against a job description using:

- Semantic similarity (Sentence Transformers)
- Skill-gap analysis
- Resume ranking
- Word-cloud visualisations

Built with Streamlit, spaCy, and sentence-transformers.

---

# Features

## Resume ↔ Job Description Comparison

Every uploaded resume is evaluated against the job description using:

| Metric | Description |
|--------|-------------|
| Overall Score | Weighted score (60% semantic similarity + 40% skill match) |
| Semantic Similarity | AI embedding similarity between JD and resume |
| Skills Match % | Percentage of JD skills found in the resume |
| Missing Skills | Skills required in JD but absent in resume |
| Word Clouds | Visual keyword comparison |

---

# Tech Stack

- Python
- Streamlit
- Sentence Transformers
- spaCy
- PyMuPDF
- scikit-learn
- WordCloud

---

# Project Structure

```text
resume-screening/
├── app.py                 ← main Streamlit application
├── requirements.txt       ← dependencies
├── runtime.txt            ← Python version
├── README.md
└── uploads/               ← temporary uploaded files