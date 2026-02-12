from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from pypdf import PdfReader
import re
import math
from collections import Counter

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

import pandas as pd

# --- DATABASES FROM CSV ---
def load_data():
    try:
        # Load Resumes
        # Columns: ID, Category, Resume
        resumes_df = pd.read_csv('UpdatedResumeDataSet.csv', sep=';')
        resumes_db = []
        for _, row in resumes_df.iterrows():
            resumes_db.append({
                "id": row['ID'],
                "name": row['Category'], # Using Category as name/title substitute
                "text": str(row['Resume'])
            })
        
        # Load Jobs
        # Columns: ID, Company, Job Title, City, Country, Job Description, url, Date
        jobs_df = pd.read_csv('jobs.csv', sep=';', on_bad_lines='skip', encoding='utf-8')
        jobs_db = []
        for _, row in jobs_df.iterrows():
            jobs_db.append({
                "id": row['ID'],
                "title": row['Job Title'],
                "description": str(row['Job Description'])
            })
            
        return jobs_db, resumes_db
    except Exception as e:
        print(f"Error loading CSV data: {e}")
        return [], []

JOBS_DB, RESUMES_DB = load_data()

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
    total_jobs = len(JOBS_DB)
    total_pages = math.ceil(total_jobs / per_page)
    
    start = (page - 1) * per_page
    end = start + per_page
    
    current_jobs = JOBS_DB[start:end]
    
    return render_template('listings.html', jobs=current_jobs, page=page, total_pages=total_pages)

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
            scored_resumes = []
            for resume in RESUMES_DB:
                score = get_cosine_similarity(job_desc_text, resume['text'])
                scored_resumes.append(resume | {"score": round(score * 100, 2)}) # Add score to resume dict
            
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
            scored_jobs = []
            for job in JOBS_DB:
                score = get_cosine_similarity(resume_text, job['description'])
                scored_jobs.append(job | {"score": round(score * 100, 2)})
            
            # Sort by score descending
            matches = sorted(scored_jobs, key=lambda x: x['score'], reverse=True)[:3]
            
            os.remove(filepath)

    return render_template('job_seeker.html', matches=matches)

if __name__ == '__main__':
    app.run(debug=True)
