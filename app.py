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
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    "pool_pre_ping": True,
    "pool_recycle": 300,
}

db = SQLAlchemy(app)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- MODELS ---
class Job(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=False)
    url = db.Column(db.Text, nullable=True) # Added URL column

class Resume(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    category = db.Column(db.String(255), nullable=False)
    resume_text = db.Column(db.Text, nullable=False)

# --- GLOBAL DATA CACHE ---
# This cache will hold data in memory to avoid repeated DB fetches and vectorization
# Structure:
# {
#   "jobs": [{"id": 1, "title": "...", "description": "...", "url": "..."}, ...],
#   "resumes": [{"id": 1, "category": "...", "text": "..."}, ...],
#   "job_vectorizer": TfidfVectorizer object (fitted on jobs),
#   "resume_vectorizer": TfidfVectorizer object (fitted on resumes),
#   "job_matrix": scipy sparse matrix (fitted on jobs),
#   "resume_matrix": scipy sparse matrix (fitted on resumes)
# }
DATA_CACHE = {
    "jobs": [],
    "resumes": [],
    "job_vectorizer": None,
    "resume_vectorizer": None,
    "job_matrix": None,
    "resume_matrix": None
}

def load_data():
    """
    Loads data from DB, cleans text, and pre-computes TF-IDF matrices.
    Called at startup.
    Uses yield_per to reduce memory usage during query.
    """
    global DATA_CACHE
    print("Loading data into cache...")
    
    with app.app_context():
        # Load Jobs
        try:
            # Use yield_per to stream results (1000 at a time) to avoid loading all into session memory
            query = Job.query.yield_per(1000)
            
            job_data = []
            job_texts = []
            
            for job in query:
                cleaned_desc = clean_text(job.description)
                job_data.append({
                    "id": job.id,
                    "title": job.title,
                    "description": job.description,
                    "url": job.url
                    # Removed cleaned_text to save memory
                })
                job_texts.append(cleaned_desc)
            
            DATA_CACHE["jobs"] = job_data
            
            if job_texts:
                vectorizer = TfidfVectorizer(preprocessor=None) # Text already cleaned
                matrix = vectorizer.fit_transform(job_texts)
                DATA_CACHE["job_vectorizer"] = vectorizer
                DATA_CACHE["job_matrix"] = matrix
                print(f"Loaded and vectorized {len(job_data)} jobs.")
                
                # Free memory immediately
                del job_texts
            else:
                print("No jobs found in DB.")

        except Exception as e:
            print(f"Error loading jobs: {e}")

        # Load Resumes
        try:
            query = Resume.query.yield_per(1000)
            resume_data = []
            resume_texts = []
            
            for resume in query:
                cleaned_text = clean_text(resume.resume_text)
                resume_data.append({
                    "id": resume.id,
                    "category": resume.category,
                    "text": resume.resume_text
                    # Removed cleaned_text
                })
                resume_texts.append(cleaned_text)
            
            DATA_CACHE["resumes"] = resume_data
            
            if resume_texts:
                vectorizer = TfidfVectorizer(preprocessor=None)
                matrix = vectorizer.fit_transform(resume_texts)
                DATA_CACHE["resume_vectorizer"] = vectorizer
                DATA_CACHE["resume_matrix"] = matrix
                print(f"Loaded and vectorized {len(resume_data)} resumes.")
                
                # Free memory
                del resume_texts
            else:
                print("No resumes found in DB.")

        except Exception as e:
            print(f"Error loading resumes: {e}")

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
            # CSV Header: ID;Company;Job Title;City;Country;Job Description;URL;Date
            jobs_df = pd.read_csv(jobs_path, sep=';', on_bad_lines='skip', encoding='utf-8')
            jobs_to_add = []
            for _, row in jobs_df.iterrows():
                # Handle potentially missing URL
                url = row['URL'] if 'URL' in row else None
                if pd.isna(url): url = None

                job = Job(
                    id=row['ID'], 
                    title=row['Job Title'],
                    description=str(row['Job Description']),
                    url=str(url) if url else None
                )
                jobs_to_add.append(job)
            
            db.session.add_all(jobs_to_add)
            db.session.commit()
            print(f"Loaded {len(jobs_to_add)} jobs.")
            # Reload cache after population
            load_data()
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
            # Reload cache after population
            load_data()
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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ... (Previous code remains the same until clean_text) ...

# --- TEXT PROCESSING & MATCHING LOGIC ---

def clean_text(text):
    """
    Robust text cleaning:
    - Removes HTML tags
    - Removes URLs
    - Removes non-alphabetic characters
    - Normalizes whitespace
    - Converts to lowercase
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)
    # Remove non-alphabetic characters (keep spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

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
    
    # We can still use DB pagination for listings as it handles limit/offset efficiently
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
            cleaned_job_desc = clean_text(job_desc_text)
            
            # Use Cached Data
            if not DATA_CACHE["resume_vectorizer"] or DATA_CACHE["resume_matrix"] is None:
                 # Fallback or reload if cache empty (though load_data called at startup)
                 load_data()
            
            if not DATA_CACHE["resumes"]:
                os.remove(filepath)
                return render_template('employer.html', matches=[])

            vectorizer = DATA_CACHE["resume_vectorizer"]
            # Transform the single new document
            job_vector = vectorizer.transform([cleaned_job_desc])
            
            # Calculate Similarity against cached matrix
            cosine_sim = cosine_similarity(job_vector, DATA_CACHE["resume_matrix"]).flatten()
            
            scored_resumes = []
            resumes = DATA_CACHE["resumes"]
            
            # Optimize: Get top k indices instead of iterating all if N is huge
            # But Python loop is fast enough for <10k. 
            # For strict speedup, use numpy argsort but let's stick to logic for now.
            
            for i, score in enumerate(cosine_sim):
                if score > 0: 
                    scored_resumes.append({
                        "id": resumes[i]["id"],
                        "name": resumes[i]["category"],
                        "text": resumes[i]["text"][:200] + "...", 
                        "score": round(score * 100, 2)
                    })
            
            matches = sorted(scored_resumes, key=lambda x: x['score'], reverse=True)[:3]
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
            cleaned_resume = clean_text(resume_text)
            
            # Use Cached Data
            if not DATA_CACHE["job_vectorizer"] or DATA_CACHE["job_matrix"] is None:
                load_data()
            
            if not DATA_CACHE["jobs"]:
                os.remove(filepath)
                return render_template('job_seeker.html', matches=[])
            
            vectorizer = DATA_CACHE["job_vectorizer"]
            resume_vector = vectorizer.transform([cleaned_resume])
            
            cosine_sim = cosine_similarity(resume_vector, DATA_CACHE["job_matrix"]).flatten()
            
            scored_jobs = []
            jobs = DATA_CACHE["jobs"]
            
            for i, score in enumerate(cosine_sim):
                 if score > 0:
                    scored_jobs.append({
                        "id": jobs[i]["id"],
                        "title": jobs[i]["title"],
                        "description": jobs[i]["description"][:200] + "...",
                        "url": jobs[i]["url"], # Include URL
                        "score": round(score * 100, 2)
                    })
            
            matches = sorted(scored_jobs, key=lambda x: x['score'], reverse=True)[:3]
            os.remove(filepath)

    return render_template('job_seeker.html', matches=matches)

if __name__ == '__main__':
    # Load data at startup
    load_data()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
