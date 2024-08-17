import os
from flask import Flask, request, render_template_string, url_for
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import logging
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import fitz  
import spacy

# Initialize the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/content/uploads'
app.config['STATIC_FOLDER'] = '/content/static'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists(app.config['STATIC_FOLDER']):
    os.makedirs(app.config['STATIC_FOLDER'])

# Initialize SpaCy model
nlp = spacy.load('en_core_web_sm')

# Set up logging
logging.basicConfig(level=logging.INFO)

# HTML template for form and results
form_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Resume Screening</title>
</head>
<body>
    <h1>Resume Screening</h1>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <label for="job_description">Job Description:</label><br>
        <textarea name="job_description" rows="5" cols="40"></textarea><br>
        <label for="resumes">Upload Resumes:</label><br>
        <input type="file" name="resumes" multiple><br>
        <input type="submit" value="Submit">
    </form>
</body>
</html>
"""

result_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Results</title>
</head>
<body>
    <h1>Results</h1>
    {% for result in results %}
        <h2>Resume {{ loop.index }}</h2>
        <p>Similarity Score: {{ result.similarity_score }}</p>
        <p>Missing Skills: {{ result.missing_skills }}</p>
        <p>Suggestions: {{ result.suggestions }}</p>
        <img src="{{ result.wordcloud_url }}" alt="Word Cloud"><br><br>
    {% endfor %}
</body>
</html>
"""

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    return render_template_string(form_html)

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if request.method == 'POST':
            logging.info("Received POST request")

            # Get job description
            job_description = request.form['job_description']

            # Process each uploaded resume file
            results = []
            for resume in request.files.getlist('resumes'):
                if not allowed_file(resume.filename):
                    return "Invalid file type", 400

                # Save resume file
                resume_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(resume.filename))
                resume.save(resume_path)
                logging.info(f"Saved resume to {resume_path}")

                # Extract text from resume file
                resume_text = None
                if resume.filename.endswith('.pdf'):
                    with fitz.open(resume_path) as pdf:
                        resume_text = ""
                        for page in pdf:
                            resume_text += page.get_text()
                else:
                    for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                        try:
                            with open(resume_path, 'r', encoding=encoding) as file:
                                resume_text = file.read()
                            break
                        except UnicodeDecodeError:
                            continue

                if resume_text is None:
                    raise ValueError("Unable to read resume file. Please ensure it is a text file in a supported encoding or a readable PDF.")

                logging.info("Read resume text")

                # Text preprocessing
                def preprocess_text(text):
                    doc = nlp(text.lower())
                    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
                    return ' '.join(tokens)

                resume_processed = preprocess_text(resume_text)
                job_description_processed = preprocess_text(job_description)
                logging.info("Processed text for resume and job description")

                # Feature extraction using BERT
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                model = BertModel.from_pretrained('bert-base-uncased')

                def extract_features(text):
                    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
                    outputs = model(**inputs)
                    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

                resume_features = extract_features(resume_processed)
                job_description_features = extract_features(job_description_processed)
                logging.info("Extracted features using BERT")

                # Calculate cosine similarity
                similarity_score = cosine_similarity(resume_features, job_description_features)[0][0]
                logging.info(f"Calculated cosine similarity: {similarity_score}")

                # Generate word cloud from resume
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(resume_processed)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                wordcloud_filename = f"wordcloud_{secure_filename(resume.filename)}.png"
                wordcloud_path = os.path.join(app.config['STATIC_FOLDER'], wordcloud_filename)
                wordcloud.to_file(wordcloud_path)
                logging.info(f"Generated word cloud and saved to {wordcloud_path}")

                # Named Entity Recognition (NER)
                resume_doc = nlp(resume_text)
                job_description_doc = nlp(job_description)

                resume_skills = {ent.text for ent in resume_doc.ents if ent.label_ == 'SKILL'}
                job_skills = {ent.text for ent in job_description_doc.ents if ent.label_ == 'SKILL'}
                logging.info("Performed Named Entity Recognition (NER)")

                # Skill gap analysis
                missing_skills = job_skills - resume_skills

                # Improvement suggestions
                suggestions = f"Consider adding these skills to your resume: {', '.join(missing_skills)}" if missing_skills else "Your resume matches the job description well."

                # Collect results
                results.append({
                    'similarity_score': similarity_score,
                    'missing_skills': ', '.join(missing_skills),
                    'suggestions': suggestions,
                    'wordcloud_url': url_for('static', filename=wordcloud_filename)
                })

            # Sort results by similarity score in descending order
            results = sorted(results, key=lambda x: x['similarity_score'], reverse=True)

            # Render result template
            return render_template_string(result_html, results=results)
    except Exception as e:
        logging.error("Error occurred", exc_info=True)
        return f"Internal Server Error: {str(e)}", 500

# Run the Flask app
if __name__ == "__main__":
    app.run()

from pyngrok import ngrok

# Authenticate ngrok
ngrok.set_auth_token("2jrQUZ2rE3uQrNLgMfEstwy7VEt_5b6tjqqd7iAnW54e62eCn")  # Replace with your ngrok auth token

# Start ngrok tunnel
public_url = ngrok.connect(5000)
print("ngrok URL:", public_url)






