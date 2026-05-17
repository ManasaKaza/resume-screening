"""
Resume Screening Tool
Flask web app that ranks uploaded resumes against a job description using
BERT embeddings + cosine similarity, skill-gap analysis, and word-cloud visuals.

Every results section compares the **Job Description ↔ Resume**:
  • Semantic similarity (BERT cosine)
  • Skills match percentage
  • Missing-skills breakdown
  • Side-by-side word clouds (JD vs résumé)
"""

import os
import re
import uuid
import logging
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, request, render_template_string, url_for
from werkzeug.utils import secure_filename
from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import fitz          # PyMuPDF
import spacy
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
STATIC_FOLDER = os.path.join(BASE_DIR, "static")

app = Flask(__name__, static_folder=STATIC_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["ALLOWED_EXTENSIONS"] = {"pdf", "txt"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

logging.info("Loading spaCy model …")
nlp = spacy.load("en_core_web_sm")

logging.info("Loading BERT tokenizer and model …")
_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
_bert_model = BertModel.from_pretrained("bert-base-uncased")
_bert_model.eval()
logging.info("All models ready.")

TECH_SKILLS: set[str] = {
    # Languages
    "python", "java", "javascript", "typescript", "sql", "r lang", "c++", "c#",
    "golang", "rust", "scala", "kotlin", "swift", "php", "ruby", "bash",
    # ML / Data
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "data analysis", "data science", "statistics",
    "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy",
    "matplotlib", "seaborn", "xgboost", "lightgbm",
    # Cloud / DevOps
    "docker", "kubernetes", "aws", "azure", "gcp", "ci/cd", "git",
    "linux", "terraform", "ansible",
    # Web
    "flask", "django", "fastapi", "react", "angular", "vue", "node.js",
    "rest api", "graphql", "html", "css",
    # Databases
    "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    # Soft skills
    "communication", "teamwork", "leadership", "problem solving",
    "project management", "agile", "scrum",
}

def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


def extract_text(file_path: str, original_filename: str) -> str:
    """Extract plain text from a PDF or text file."""
    if original_filename.lower().endswith(".pdf"):
        text_parts = []
        with fitz.open(file_path) as pdf:
            for page in pdf:
                text_parts.append(page.get_text())
        return "\n".join(text_parts)

    for encoding in ("utf-8", "latin-1", "iso-8859-1", "cp1252"):
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, LookupError):
            continue

    raise ValueError(
        f"Could not decode '{original_filename}'. "
        "Please save it as UTF-8 or upload a PDF instead."
    )


def preprocess_for_wordcloud(text: str) -> str:
    """Lowercase, lemmatise, remove stopwords and punctuation via spaCy.
    Used ONLY for word-cloud generation — NOT for BERT embeddings."""
    doc = nlp(text.lower())
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and token.lemma_.strip()
    ]
    return " ".join(tokens)


def clean_text_for_bert(text: str) -> str:
    """Light cleanup for BERT: collapse whitespace, strip artefacts.
    BERT works best on natural language, NOT lemmatised bags of words."""
    text = re.sub(r"\s+", " ", text)    
    text = re.sub(r"[^\x20-\x7E]+", " ", text) 
    return text.strip()[:5000]  


def bert_embed(text: str):
    """Return a mean-pooled BERT embedding as a numpy array."""
    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )
    with torch.no_grad():
        outputs = _bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()


def extract_skills(text: str) -> set[str]:
    """Word-boundary-aware skill extraction to avoid false positives
    like 'go' matching inside 'google' or 'r' inside 'experience'."""
    text_lower = text.lower()
    found = set()
    for skill in TECH_SKILLS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found.add(skill)
    return found


def make_wordcloud(text: str, base_name: str) -> str | None:
    """
    Generate a word-cloud PNG, save it to the static folder.
    Returns the filename (not the full path) for use with url_for().
    Returns None if the text is empty (no words to plot).
    """
    if not text or not text.strip():
        return None

    wc = WordCloud(width=800, height=400, background_color="white")
    try:
        wc.generate(text)
    except ValueError:
        return None

    out_filename = f"wc_{uuid.uuid4().hex}_{secure_filename(base_name)}.png"
    wc.to_file(os.path.join(STATIC_FOLDER, out_filename))
    return out_filename


def compute_skills_match(jd_skills: set[str], resume_skills: set[str]) -> float:
    """Return the percentage (0-100) of JD skills found in the resume."""
    if not jd_skills:
        return 100.0
    return round(len(jd_skills & resume_skills) / len(jd_skills) * 100, 1)


def compute_overall_score(semantic_sim: float, skills_pct: float) -> float:
    """Weighted combination: 60% semantic, 40% skills match."""
    return round(semantic_sim * 0.6 + (skills_pct / 100) * 0.4, 4)


def cleanup_file(path: str) -> None:
    """Remove a file if it exists (best-effort)."""
    try:
        os.remove(path)
    except OSError:
        pass

FORM_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Resume Screening Tool</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            min-height: 100vh; display: flex; align-items: center; justify-content: center;
            color: #e0e0e0; padding: 24px;
        }
        .container {
            background: rgba(255,255,255,0.06);
            backdrop-filter: blur(18px);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 20px; padding: 48px 40px; max-width: 700px; width: 100%;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        }
        h1 {
            font-size: 2em; font-weight: 700; margin-bottom: 8px;
            background: linear-gradient(90deg, #a78bfa, #60a5fa);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .subtitle { color: #9ca3af; margin-bottom: 32px; font-size: 0.95em; }
        label { font-weight: 600; display: block; margin-bottom: 8px; color: #c4b5fd; }
        textarea {
            width: 100%; border-radius: 12px; border: 1px solid rgba(255,255,255,0.15);
            background: rgba(255,255,255,0.05); color: #e0e0e0; padding: 14px;
            font-size: 0.95em; resize: vertical; min-height: 140px;
            transition: border-color 0.3s;
        }
        textarea:focus { outline: none; border-color: #a78bfa; }
        .file-input-wrapper {
            margin-top: 24px; padding: 24px; border: 2px dashed rgba(255,255,255,0.15);
            border-radius: 12px; text-align: center; cursor: pointer;
            transition: border-color 0.3s, background 0.3s;
        }
        .file-input-wrapper:hover { border-color: #a78bfa; background: rgba(167,139,250,0.05); }
        input[type="file"] { margin-top: 8px; }
        .btn {
            margin-top: 28px; padding: 14px 36px; width: 100%;
            background: linear-gradient(135deg, #a78bfa, #6366f1);
            color: #fff; border: none; border-radius: 12px; cursor: pointer;
            font-size: 1.05em; font-weight: 600; letter-spacing: 0.5px;
            transition: transform 0.2s, box-shadow 0.3s;
        }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(99,102,241,0.4); }
        .btn:active { transform: translateY(0); }
    </style>
</head>
<body>
    <div class="container">
        <h1>Resume Screening Tool</h1>
        <p class="subtitle">Upload resumes and compare them against a job description using AI-powered analysis.</p>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <label for="job_description">Job Description</label>
            <textarea id="job_description" name="job_description" rows="8"
                      placeholder="Paste the full job description here…" required></textarea>

            <div class="file-input-wrapper">
                <label for="resumes">Upload Resumes (PDF or TXT, multiple allowed)</label>
                <input type="file" id="resumes" name="resumes" multiple accept=".pdf,.txt" required>
            </div>

            <button class="btn" type="submit">Screen Resumes</button>
        </form>
    </div>
</body>
</html>
"""

RESULT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Screening Results</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            min-height: 100vh; color: #e0e0e0; padding: 32px 24px;
        }
        .wrapper { max-width: 900px; margin: 0 auto; }
        h1 {
            font-size: 2em; font-weight: 700; margin-bottom: 6px;
            background: linear-gradient(90deg, #a78bfa, #60a5fa);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .subtitle { color: #9ca3af; margin-bottom: 28px; }

        /* JD overview card */
        .jd-card {
            background: rgba(255,255,255,0.06); backdrop-filter: blur(16px);
            border: 1px solid rgba(255,255,255,0.12); border-radius: 16px;
            padding: 28px; margin-bottom: 36px;
        }
        .jd-card h2 { color: #c4b5fd; margin-bottom: 12px; font-size: 1.2em; }
        .skill-tag {
            display: inline-block; padding: 4px 12px; margin: 3px 4px;
            background: rgba(99,102,241,0.25); border-radius: 20px;
            font-size: 0.82em; color: #a5b4fc;
        }

        /* Resume card */
        .card {
            background: rgba(255,255,255,0.06); backdrop-filter: blur(16px);
            border: 1px solid rgba(255,255,255,0.12); border-radius: 16px;
            padding: 28px; margin-bottom: 28px;
            transition: transform 0.2s, box-shadow 0.3s;
        }
        .card:hover { transform: translateY(-3px); box-shadow: 0 8px 28px rgba(0,0,0,0.3); }
        .card h2 { color: #e0e0e0; margin-bottom: 16px; font-size: 1.15em; }

        /* Score grid */
        .score-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 20px; }
        .score-box {
            background: rgba(255,255,255,0.04); border-radius: 12px; padding: 16px; text-align: center;
        }
        .score-box .label { font-size: 0.78em; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.5px; }
        .score-box .value { font-size: 1.6em; font-weight: 700; margin-top: 4px; }
        .score-box .value.high   { color: #34d399; }
        .score-box .value.medium { color: #fbbf24; }
        .score-box .value.low    { color: #f87171; }

        .missing-skills { margin: 12px 0; }
        .missing-tag {
            display: inline-block; padding: 4px 12px; margin: 3px 4px;
            background: rgba(248,113,113,0.2); border-radius: 20px;
            font-size: 0.82em; color: #fca5a5;
        }
        .matched-tag {
            display: inline-block; padding: 4px 12px; margin: 3px 4px;
            background: rgba(52,211,153,0.2); border-radius: 20px;
            font-size: 0.82em; color: #6ee7b7;
        }

        .suggestion { color: #9ca3af; font-style: italic; margin: 12px 0; }

        .wc-row { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 16px; }
        .wc-col { flex: 1; min-width: 250px; }
        .wc-col h4 { font-size: 0.85em; color: #9ca3af; margin-bottom: 8px; }
        img {
            max-width: 100%; border-radius: 10px; border: 1px solid rgba(255,255,255,0.08);
        }

        a.back {
            display: inline-block; margin-top: 28px; color: #a78bfa;
            text-decoration: none; font-weight: 600;
            transition: color 0.2s;
        }
        a.back:hover { color: #c4b5fd; }

        @media (max-width: 600px) {
            .score-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="wrapper">
        <h1>Screening Results</h1>
        <!-- JD overview -->
        <div class="jd-card">
            <h2>📋 Job Description Skills Detected</h2>
            {% if jd_skills %}
                {% for s in jd_skills|sort %}
                    <span class="skill-tag">{{ s }}</span>
                {% endfor %}
            {% else %}
                <p style="color:#9ca3af;">No specific skills detected in the job description.</p>
            {% endif %}
            {% if jd_wc_url %}
                <h4 style="margin-top:16px; font-size:0.85em; color:#9ca3af;">JD Word Cloud</h4>
                <img src="{{ jd_wc_url }}" alt="Job description word cloud" style="margin-top:8px; max-width:60%;">
            {% endif %}
        </div>

        {% for r in results %}
        <div class="card">
            <h2>#{{ loop.index }} — {{ r.filename }}</h2>

            <!-- Score breakdown: JD vs Resume -->
            <div class="score-grid">
                <div class="score-box">
                    <div class="label">Overall Score</div>
                    <div class="value {% if r.overall_score >= 0.7 %}high{% elif r.overall_score >= 0.4 %}medium{% else %}low{% endif %}">
                        {{ "%.1f"|format(r.overall_score * 100) }}%
                    </div>
                </div>
                <div class="score-box">
                    <div class="label">Semantic Similarity</div>
                    <div class="value {% if r.semantic_score >= 0.7 %}high{% elif r.semantic_score >= 0.4 %}medium{% else %}low{% endif %}">
                        {{ "%.1f"|format(r.semantic_score * 100) }}%
                    </div>
                </div>
                <div class="score-box">
                    <div class="label">Skills Match</div>
                    <div class="value {% if r.skills_pct >= 70 %}high{% elif r.skills_pct >= 40 %}medium{% else %}low{% endif %}">
                        {{ r.skills_pct }}%
                    </div>
                </div>
            </div>

            <!-- Matched skills -->
            {% if r.matched_skills %}
            <p><strong style="color:#6ee7b7;">Matched Skills:</strong></p>
            <div style="margin-bottom:8px;">
                {% for s in r.matched_skills %}
                    <span class="matched-tag">{{ s }}</span>
                {% endfor %}
            </div>
            {% endif %}

            <!-- Missing skills -->
            {% if r.missing_skills_list %}
            <p><strong style="color:#fca5a5;">Missing Skills (in JD but not in resume):</strong></p>
            <div class="missing-skills">
                {% for s in r.missing_skills_list %}
                    <span class="missing-tag">{{ s }}</span>
                {% endfor %}
            </div>
            {% endif %}

            <p class="suggestion">{{ r.suggestion }}</p>

            <!-- Side-by-side word clouds: JD vs Resume -->
            <div class="wc-row">
                {% if jd_wc_url %}
                <div class="wc-col">
                    <h4>JD Word Cloud</h4>
                    <img src="{{ jd_wc_url }}" alt="JD word cloud">
                </div>
                {% endif %}
                {% if r.resume_wc_url %}
                <div class="wc-col">
                    <h4>Resume Word Cloud</h4>
                    <img src="{{ r.resume_wc_url }}" alt="Resume word cloud for {{ r.filename }}">
                </div>
                {% endif %}
            </div>
        </div>
        {% endfor %}

        <a class="back" href="/">← Screen more resumes</a>
    </div>
</body>
</html>
"""

ERROR_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Error</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            min-height: 100vh; display: flex; align-items: center; justify-content: center;
            color: #e0e0e0; padding: 24px;
        }
        .error-card {
            background: rgba(248,113,113,0.08); backdrop-filter: blur(16px);
            border: 1px solid rgba(248,113,113,0.25); border-radius: 16px;
            padding: 40px; max-width: 520px; text-align: center;
        }
        h1 { color: #f87171; margin-bottom: 16px; }
        p { color: #d1d5db; line-height: 1.6; }
        a { color: #a78bfa; display: inline-block; margin-top: 20px; text-decoration: none; font-weight: 600; }
        a:hover { color: #c4b5fd; }
    </style>
</head>
<body>
    <div class="error-card">
        <h1>Error</h1>
        <p>{{ message }}</p>
        <a href="/">Go back</a>
    </div>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(FORM_HTML)


@app.route("/upload", methods=["POST"])
def upload_file():
    saved_paths = []  
    try:
        job_description = request.form.get("job_description", "").strip()
        if not job_description:
            return render_template_string(ERROR_HTML,
                                          message="Job description is required."), 400

        resume_files = request.files.getlist("resumes")
        valid_files = [f for f in resume_files if f and f.filename]
        if not valid_files:
            return render_template_string(ERROR_HTML,
                                          message="Please upload at least one resume."), 400

        jd_clean    = clean_text_for_bert(job_description)
        jd_features = bert_embed(jd_clean)
        jd_skills   = extract_skills(job_description)

        jd_wc_text = preprocess_for_wordcloud(job_description)
        jd_wc_file = make_wordcloud(jd_wc_text, "job_description")
        jd_wc_url  = url_for("static", filename=jd_wc_file) if jd_wc_file else None

        results = []

        for resume in valid_files:
            original_name = resume.filename

            if not allowed_file(original_name):
                return render_template_string(
                    ERROR_HTML,
                    message=f"Invalid file type for '{original_name}'. "
                            "Only PDF and TXT files are accepted."
                ), 400

            safe_name   = f"{uuid.uuid4().hex[:8]}_{secure_filename(original_name)}"
            resume_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
            resume.save(resume_path)
            saved_paths.append(resume_path)
            logging.info("Saved: %s", resume_path)

            resume_text = extract_text(resume_path, original_name)

            resume_clean    = clean_text_for_bert(resume_text)
            resume_features = bert_embed(resume_clean)
            semantic_score  = float(
                cosine_similarity(resume_features, jd_features)[0][0]
            )
            logging.info("%s → semantic similarity %.4f", safe_name, semantic_score)

            resume_skills  = extract_skills(resume_text)
            matched_skills = sorted(jd_skills & resume_skills)
            missing_skills = sorted(jd_skills - resume_skills)
            skills_pct     = compute_skills_match(jd_skills, resume_skills)

            overall_score = compute_overall_score(semantic_score, skills_pct)

            resume_wc_text = preprocess_for_wordcloud(resume_text)
            resume_wc_file = make_wordcloud(resume_wc_text, safe_name)
            resume_wc_url  = (
                url_for("static", filename=resume_wc_file) if resume_wc_file else None
            )

            suggestion = (
                f"Consider highlighting these skills: {', '.join(missing_skills)}"
                if missing_skills
                else "Great match! Your resume aligns well with the job description."
            )

            results.append({
                "filename":           original_name,
                "semantic_score":     semantic_score,
                "skills_pct":         skills_pct,
                "overall_score":      overall_score,
                "matched_skills":     matched_skills,
                "missing_skills_list": missing_skills,
                "suggestion":         suggestion,
                "resume_wc_url":      resume_wc_url,
            })

        results.sort(key=lambda x: x["overall_score"], reverse=True)

        return render_template_string(
            RESULT_HTML,
            results=results,
            jd_skills=jd_skills,
            jd_wc_url=jd_wc_url,
        )

    except Exception as exc:
        logging.error("Unhandled error", exc_info=True)
        return render_template_string(ERROR_HTML, message=str(exc)), 500

    finally:
        for path in saved_paths:
            cleanup_file(path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)
