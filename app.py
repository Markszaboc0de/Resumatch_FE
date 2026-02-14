from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from pypdf import PdfReader
import re
import math
from collections import Counter
from flask_sqlalchemy import SQLAlchemy
import pandas as pd

# Define base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = "postgresql://neondb_owner:npg_OBP96fcFuQYA@ep-quiet-field-agaxzybz-pooler.c-2.eu-central-1.aws.neon.tech/neondb?sslmode=require"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- MODELS ---
class Job(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=False)

class Resume(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    category = db.Column(db.String(255), nullable=False)
    resume_text = db.Column(db.Text, nullable=False)

# --- DATABASE MIGRATION ---
def populate_jobs(clear=False):
    try:
        if clear:
            db.session.query(Job).delete()
            db.session.commit()
            print("Cleared Jobs table.")

        jobs_path = os.path.join(BASE_DIR, 'jobs.csv')
        
        if os.path.exists(jobs_path):
            print("Loading Jobs from CSV...")
            jobs_df = pd.read_csv(jobs_path, sep=';', on_bad_lines='skip', encoding='utf-8')
            jobs_to_add = []
            for _, row in jobs_df.iterrows():
                job = Job(
                    id=row['ID'], 
                    title=row['Job Title'],
                    description=str(row['Job Description'])
                )
                jobs_to_add.append(job)
            
            db.session.add_all(jobs_to_add)
            db.session.commit()
            print(f"Loaded {len(jobs_to_add)} jobs.")
        else:
            print("No jobs CSV found.")

    except Exception as e:
        db.session.rollback()
        print(f"Error loading jobs: {e}")

def populate_resumes(clear=False):
    try:
        if clear:
            db.session.query(Resume).delete()
            db.session.commit()
            print("Cleared Resumes table.")

        resumes_path = os.path.join(BASE_DIR, 'UpdatedResumeDataSet.csv')
        if os.path.exists(resumes_path):
            print("Loading Resumes from CSV...")
            resumes_df = pd.read_csv(resumes_path, sep=';')
            resumes_to_add = []
            for _, row in resumes_df.iterrows():
                resume = Resume(
                    category=row['Category'],
                    resume_text=str(row['Resume'])
                )
                resumes_to_add.append(resume)
            db.session.add_all(resumes_to_add)
            db.session.commit()
            print(f"Loaded {len(resumes_to_add)} resumes.")
        else:
            print("No resumes CSV found.")
    except Exception as e:
        db.session.rollback()
        print(f"Error loading resumes: {e}")

def init_db():
    with app.app_context():
        db.create_all()
        # Data population is now handled manually via manage_data.py
        # to prevent app from trying to read CSVs on deployment

# Initialize DB (This will run on import, effectively checking/migrating on startup)
# In production specifically, you might want this in a separate script.
# For this task, we'll call it before run.
init_db()


# --- TEXT PROCESSING & MATCHING LOGIC ---

def clean_text(text):
    """
    Tokenizes and cleans text: removes non-alphanumeric characters, converts to lower case,
    and removes common stopwords.
    Returns a list of words.
    """
    STOPWORDS = {
        'and', 'the', 'is', 'in', 'at', 'of', 'a', 'with', 'using', 'for', 'to', 'an', 'or', 'by', 'on'
    }
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = text.lower().split()
    return [word for word in tokens if word not in STOPWORDS]

def get_cosine_similarity(text1, text2):
    """
    Calculates Cosine Similarity between two text strings.
    """
    tokens1 = clean_text(text1)
    tokens2 = clean_text(text2)

    vec1 = Counter(tokens1)
    vec2 = Counter(tokens2)

    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    return numerator / denominator

def extract_text_from_pdf(filepath):
    """
    Extracts text from a PDF file using pypdf.
    """
    try:
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def extract_text_from_txt(filepath):
    """
    Extracts text from a TXT file.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading TXT: {e}")
        return ""

# --- ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/listings')
def listings():
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    pagination = Job.query.paginate(page=page, per_page=per_page, error_out=False)
    current_jobs = pagination.items
    
    return render_template('listings.html', jobs=current_jobs, page=page, total_pages=pagination.pages, has_next=pagination.has_next, has_prev=pagination.has_prev)

@app.route('/employer', methods=['GET', 'POST'])
def employer():
    matches = []
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file and file.filename.endswith('.txt'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            job_desc_text = extract_text_from_txt(filepath)
            
            # Match against Resumes Database
            # Note: Fetching all resumes into memory might be heavy if DB is huge.
            # Ideally, we'd do this in the DB, but PostgreSQL Full Text Search or pgvector is better for this.
            # For now, we fetch all (users said "database got so big", so this might be a bottleneck later,
            # but for this step we are just moving storage).
            all_resumes = Resume.query.all()
            
            scored_resumes = []
            for resume in all_resumes:
                score = get_cosine_similarity(job_desc_text, resume.resume_text)
                scored_resumes.append({
                    "id": resume.id,
                    "name": resume.category,
                    "text": resume.resume_text,
                    "score": round(score * 100, 2)
                })
            
            # Sort by score descending
            matches = sorted(scored_resumes, key=lambda x: x['score'], reverse=True)[:3]
            
            # Simple cleanup of uploaded file
            os.remove(filepath)
            
    return render_template('employer.html', matches=matches)

@app.route('/job_seeker', methods=['GET', 'POST'])
def job_seeker():
    matches = []
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file and file.filename.endswith('.pdf'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            resume_text = extract_text_from_pdf(filepath)
            
            # Match against Jobs Database
            all_jobs = Job.query.all()
            
            scored_jobs = []
            for job in all_jobs:
                score = get_cosine_similarity(resume_text, job.description)
                scored_jobs.append({
                    "id": job.id,
                    "title": job.title,
                    "description": job.description,
                    "score": round(score * 100, 2)
                })
            
            # Sort by score descending
            matches = sorted(scored_jobs, key=lambda x: x['score'], reverse=True)[:3]
            
            os.remove(filepath)

    return render_template('job_seeker.html', matches=matches)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

