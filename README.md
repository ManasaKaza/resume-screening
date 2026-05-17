# Resume Screening Tool

AI-powered resume screening tool that ranks uploaded resumes against a job description using **BERT semantic embeddings**, **skill-gap analysis**, and **word-cloud visualisations**.

## What it does

Every results section **compares JD ↔ Resume**:

| Metric | How it works |
|--------|-------------|
| **Overall Score** | Weighted combination (60% semantic + 40% skills) |
| **Semantic Similarity** | BERT cosine similarity between JD and resume text |
| **Skills Match %** | Percentage of JD-required skills found in the resume |
| **Matched / Missing Skills** | Colour-coded tags showing which JD skills are present or absent |
| **Word Clouds** | Side-by-side JD vs resume word clouds for visual comparison |

---

## Project Structure

```
resume-screening/
├── resume_screening.py   ← main Flask app
├── requirements.txt      ← dependencies (including spaCy model)
├── Procfile              ← for Heroku / Render (gunicorn)
├── runtime.txt           ← Python version for deploy platforms
├── .gitignore            ← excludes generated files
├── uploads/              ← auto-created at runtime (cleaned after each request)
└── static/               ← auto-created, stores word-cloud PNGs
```

---

## Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.10, 3.11, or 3.12 |
| pip | latest |
| Git | any |

---

## How to Test

1. Open http://127.0.0.1:5000
2. Paste a job description (e.g. *"Python developer with machine learning, Docker, and AWS experience"*)
3. Upload one or more `.pdf` or `.txt` résumé files
4. Click **Screen Resumes**
5. Results page shows for **each resume compared against the JD**:
   - Overall / Semantic / Skills Match scores
   - Matched skills (green) and missing skills (red)
   - Side-by-side word clouds