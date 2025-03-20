import fitz  # PyMuPDF
import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from concurrent.futures import ThreadPoolExecutor
import sqlite3

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Predefined skills set
SKILLS_SET = {
    "data analysis", "data visualization", "data mining", "data cleaning", "data wrangling",
    "python", "r", "sql", "excel", "tableau", "power bi", "machine learning",
    "statistical analysis", "predictive modeling", "big data", "aws", "cloud computing",
    "docker", "git", "communication", "project management", "linux"
}

# Precompiled regex patterns
CGPA_PATTERN = re.compile(r'\b(?:CGPA|GPA)\s*:?[\s-]([0-9]+(?:\.[0-9]+)?)\b', re.IGNORECASE)
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', re.IGNORECASE)
EXPERIENCE_PATTERN = re.compile(r'(\d+)\s*(?:year|yr|y)\b', re.IGNORECASE)

# Load SBERT model
MODEL = SentenceTransformer('all-MiniLM-L6-v2')

# Text extraction with debugging
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file with error handling."""
    try:
        doc = fitz.open(pdf_path)
        text = "".join(page.get_text() for page in doc)
        doc.close()
        print(f"Extracted text from {pdf_path}: {text[:100]}...")  # Debug: Show first 100 chars
        return text if text else "No text extracted"
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return "Error: Unable to read PDF"

# Preprocess text with fallback
def preprocess_text(text, stop_words=set(stopwords.words('english'))):
    """Cleans and tokenizes text with fallback."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = [word for word in word_tokenize(text) if word not in stop_words]
    processed = ' '.join(tokens) if tokens else text
    print(f"Preprocessed text sample: {processed[:100]}...")  # Debug
    return processed

# Feature extraction
def extract_features(text):
    """Extracts CGPA, email, skills, and experience."""
    cgpa = CGPA_PATTERN.search(text)
    email = EMAIL_PATTERN.search(text)
    experience = EXPERIENCE_PATTERN.search(text)
    skills = {skill for skill in SKILLS_SET if skill in text.lower()}
    features = {
        "cgpa": float(cgpa.group(1)) if cgpa else None,
        "email": email.group(0) if email else "Not found",
        "experience": int(experience.group(1)) if experience else 0,
        "skills": skills
    }
    print(f"Extracted features: CGPA={features['cgpa']}, Skills={features['skills']}")  # Debug
    return features

# Compute weighted score with relaxed defaults
def compute_weighted_score(similarity, skill_match_ratio, cgpa=None, experience=0,
                          weights={'similarity': 0.4, 'skills': 0.4, 'cgpa': 0.1, 'experience': 0.1}):
    """Combines features into a weighted score."""
    skill_score = skill_match_ratio
    cgpa_score = min(cgpa / 4.0, 1.0) if cgpa else 0.0
    exp_score = min(experience / 10, 1.0) if experience else 0.0
    score = (weights['similarity'] * similarity +
             weights['skills'] * skill_score +
             weights['cgpa'] * cgpa_score +
             weights['experience'] * exp_score) * 100
    print(f"Score components: Sim={similarity:.2f}, Skills={skill_score:.2f}, CGPA={cgpa_score:.2f}, Exp={exp_score:.2f}, Total={score:.2f}")  # Debug
    return score

# Main execution
if __name__ == "__main__":
    # File paths (ensure these are correct)
    job_desc_path = "./job_description.pdf"
    resume_paths = [
        "C:/Users/ASUS/Downloads/Mugesh Rao CV.pdf",
        "C:/Users/ASUS/Desktop/Resume - Elliot.pdf",
        "C:/Users/ASUS/Downloads/Resume Template.pdf"
    ]

    # Extract job description
    job_text_raw = extract_text_from_pdf(job_desc_path)
    if not job_text_raw or "Error" in job_text_raw:
        print("Fatal Error: Job description could not be read. Exiting.")
        exit(1)
    job_text = preprocess_text(job_text_raw)
    job_features = extract_features(job_text_raw)
    job_skills = job_features["skills"]
    total_job_skills = max(len(job_skills), 1)

    # Batch process resumes
    with ThreadPoolExecutor() as executor:
        resume_texts_raw = list(executor.map(extract_text_from_pdf, resume_paths))
    resume_texts = [preprocess_text(text) for text in resume_texts_raw]
    resume_features = [extract_features(text) for text in resume_texts_raw]

    # Generate embeddings and compute similarity
    all_texts = [job_text] + resume_texts
    embeddings = MODEL.encode(all_texts, show_progress_bar=False, batch_size=32, convert_to_tensor=True)
    job_embedding = embeddings[0]
    resume_embeddings = embeddings[1:]
    similarities = util.cos_sim(job_embedding, resume_embeddings).cpu().numpy().flatten()
    print(f"Similarities: {similarities}")  # Debug

    # Process results
    results = []
    for i, (resume_path, features, similarity) in enumerate(zip(resume_paths, resume_features, similarities)):
        skill_matches = len(job_skills & features["skills"])
        skill_match_ratio = skill_matches / total_job_skills
        score = compute_weighted_score(similarity, skill_match_ratio, features["cgpa"], features["experience"])

        results.append({
            "Resume": resume_path.split("/")[-1],
            "Match Score (%)": round(score, 2),
            "Similarity (%)": round(similarity * 100, 2),
            "Skill Matches": skill_matches,
            "Matched Skills": ", ".join(features["skills"]) or "None",
            "Total Job Skills": total_job_skills,
            "CGPA": features["cgpa"] if features["cgpa"] else "Not found",
            "Experience (Years)": features["experience"],
            "Email": features["email"]
        })

    # Create DataFrame and filter
    df = pd.DataFrame(results)
    filtered_df = df[df["Match Score (%)"] >= 50].sort_values("Match Score (%)", ascending=False)  # Lowered to 50%

    # Save to SQLite
    conn = sqlite3.connect("resume_matches.db")
    df.to_sql("candidates", conn, if_exists="replace", index=False)
    conn.close()

    # Display results
    print("Job Skills identified:", sorted(job_skills))
    print("\nFull Results:")
    print(df)
    print("\nFiltered Candidates (Score >= 50%):")
    if filtered_df.empty:
        print("No candidates met the 50% threshold. Check data or lower threshold.")
    else:
        print(filtered_df)
    print("\nFull results saved to 'resume_matches.db'")