import os
import re
import uuid
import logging
import tempfile

import fitz  # PyMuPDF
import spacy
import torch
import matplotlib

matplotlib.use("Agg")

from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import streamlit as st

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="AI Resume Screening Tool",
    layout="wide",
)

TECH_SKILLS = {
    # Languages
    "python", "java", "javascript", "typescript", "sql",
    "c++", "c#", "golang", "rust", "scala", "kotlin",
    "swift", "php", "ruby", "bash",

    # ML / Data
    "machine learning", "deep learning", "nlp",
    "natural language processing", "computer vision",
    "data analysis", "data science", "statistics",
    "tensorflow", "pytorch", "keras", "scikit-learn",
    "pandas", "numpy", "matplotlib", "xgboost",

    # Cloud / DevOps
    "docker", "kubernetes", "aws", "azure", "gcp",
    "ci/cd", "git", "linux", "terraform",

    # Web
    "flask", "django", "fastapi", "react",
    "angular", "vue", "node.js",
    "rest api", "graphql", "html", "css",

    # Databases
    "mysql", "postgresql", "mongodb",
    "redis", "elasticsearch",

    # Soft skills
    "communication", "teamwork", "leadership",
    "problem solving", "project management",
    "agile", "scrum",
}

@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


nlp = load_spacy()
model = load_model()

def extract_text(uploaded_file):
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        text_parts = []

        with fitz.open(tmp_path) as pdf:
            for page in pdf:
                text_parts.append(page.get_text())

        os.remove(tmp_path)

        return "\n".join(text_parts)

    else:
        return uploaded_file.read().decode("utf-8", errors="ignore")


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x20-\x7E]+", " ", text)
    return text.strip()


def preprocess_for_wordcloud(text):
    doc = nlp(text.lower())

    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and token.lemma_.strip()
    ]

    return " ".join(tokens)


def extract_skills(text):
    text_lower = text.lower()
    found = set()

    for skill in TECH_SKILLS:
        pattern = r"\b" + re.escape(skill) + r"\b"

        if re.search(pattern, text_lower):
            found.add(skill)

    return found


def get_embedding(text):
    return model.encode([text])


def semantic_similarity(jd_text, resume_text):
    jd_emb = get_embedding(jd_text)
    resume_emb = get_embedding(resume_text)

    score = cosine_similarity(jd_emb, resume_emb)[0][0]

    return float(score)


def skills_match(jd_skills, resume_skills):
    if not jd_skills:
        return 100.0

    return round(
        len(jd_skills & resume_skills) / len(jd_skills) * 100,
        1
    )


def overall_score(semantic_score, skills_pct):
    return round(
        semantic_score * 0.6 + (skills_pct / 100) * 0.4,
        4
    )


def create_wordcloud(text):
    if not text.strip():
        return None

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white"
    )

    return wc.generate(text)

st.title("AI Resume Screening Tool")

st.markdown(
    """
    Upload resumes and compare them against a job description using:

    - Semantic similarity
    - Skill matching
    - Missing skills analysis
    - AI-powered ranking
    """
)

job_description = st.text_area(
    "Paste Job Description",
    height=250
)

uploaded_files = st.file_uploader(
    "Upload Resumes (PDF or TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if st.button("Screen Resumes"):

    if not job_description.strip():
        st.error("Please enter a job description.")
        st.stop()

    if not uploaded_files:
        st.error("Please upload at least one resume.")
        st.stop()

    with st.spinner("Analyzing resumes..."):

        jd_clean = clean_text(job_description)

        jd_skills = extract_skills(job_description)

        jd_wc_text = preprocess_for_wordcloud(job_description)

        jd_wordcloud = create_wordcloud(jd_wc_text)

        results = []

        for uploaded_file in uploaded_files:

            try:
                resume_text = extract_text(uploaded_file)

                resume_clean = clean_text(resume_text)

                semantic_score = semantic_similarity(
                    jd_clean,
                    resume_clean
                )

                resume_skills = extract_skills(resume_text)

                matched_skills = sorted(
                    jd_skills & resume_skills
                )

                missing_skills = sorted(
                    jd_skills - resume_skills
                )

                skills_pct = skills_match(
                    jd_skills,
                    resume_skills
                )

                final_score = overall_score(
                    semantic_score,
                    skills_pct
                )

                resume_wc_text = preprocess_for_wordcloud(
                    resume_text
                )

                resume_wordcloud = create_wordcloud(
                    resume_wc_text
                )

                results.append({
                    "filename": uploaded_file.name,
                    "semantic_score": semantic_score,
                    "skills_pct": skills_pct,
                    "overall_score": final_score,
                    "matched_skills": matched_skills,
                    "missing_skills": missing_skills,
                    "resume_wordcloud": resume_wordcloud,
                })

            except Exception as e:
                st.error(
                    f"Error processing {uploaded_file.name}: {str(e)}"
                )

        results.sort(
            key=lambda x: x["overall_score"],
            reverse=True
        )

    st.success("Analysis Complete!")

    st.header("Job Description Skills")

    if jd_skills:
        st.write(", ".join(sorted(jd_skills)))
    else:
        st.info("No skills detected.")

    if jd_wordcloud:
        st.subheader("JD Word Cloud")
        st.image(jd_wordcloud.to_array())

    st.header("Ranked Results")

    for idx, result in enumerate(results, start=1):

        st.markdown("---")

        st.subheader(f"#{idx} - {result['filename']}")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Overall Score",
                f"{result['overall_score'] * 100:.1f}%"
            )

        with col2:
            st.metric(
                "Semantic Similarity",
                f"{result['semantic_score'] * 100:.1f}%"
            )

        with col3:
            st.metric(
                "Skills Match",
                f"{result['skills_pct']}%"
            )

        st.markdown("Matched Skills")

        if result["matched_skills"]:
            st.write(", ".join(result["matched_skills"]))
        else:
            st.write("No matched skills.")

        st.markdown("Missing Skills")

        if result["missing_skills"]:
            st.write(", ".join(result["missing_skills"]))
        else:
            st.write("No missing skills.")

        if result["resume_wordcloud"]:
            st.markdown("Resume Word Cloud")
            st.image(result["resume_wordcloud"].to_array())