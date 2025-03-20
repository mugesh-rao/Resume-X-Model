import fitz  # PyMuPDF
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from functools import lru_cache  # For memoization

# Download only necessary NLTK data (run once)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Predefined list of skills (domain-specific)
SKILLS_LIST = {
    "data analysis", "data visualization", "data mining", "data cleaning", "data wrangling",
    "python", "r", "sql", "excel", "tableau", "power bi", "machine learning",
    "statistical analysis", "predictive modeling", "big data", "aws", "cloud computing",
    "docker", "git", "communication", "project management", "linux"
}

# Precompile regex patterns for efficiency
CGPA_PATTERN = re.compile(r'\b(?:CGPA|GPA)\s*:?[\s-]([0-9]+(?:\.[0-9]+)?)\b', re.IGNORECASE)
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', re.IGNORECASE)

# Memoized text extraction to avoid redundant file reads
@lru_cache(maxsize=128)
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file (cached)."""
    doc = fitz.open(pdf_path)
    text = "".join(page.get_text() for page in doc)
    doc.close()  # Explicitly close to free resources
    return text

# Preprocess text efficiently
def preprocess_text(text, stop_words=set(stopwords.words('english'))):
    """Cleans and tokenizes text in one pass."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = [word for word in word_tokenize(text) if word not in stop_words]
    return ' '.join(tokens)

# Feature extraction functions
def extract_cgpa(text):
    """Extracts CGPA using precompiled regex."""
    match = CGPA_PATTERN.search(text)
    return float(match.group(1)) if match else None

def extract_email(text):
    """Extracts email using precompiled regex."""
    match = EMAIL_PATTERN.search(text)
    return match.group(0) if match else "Not found"

def extract_skills(text, skills_set=SKILLS_LIST):
    """Extracts skills efficiently using set intersection."""
    text_lower = text.lower()
    return [skill for skill in skills_set if skill in text_lower]  # Simplified check

# Optimized TF-IDF similarity calculation
def calculate_similarity(job_text, resume_texts, max_features=1000):
    """Calculates TF-IDF cosine similarity efficiently."""
    all_texts = [job_text] + resume_texts
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features, 
                                 lowercase=False)  # Already lowercased
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return similarities

# Weighted score computation
def compute_weighted_score(similarity, skill_match_ratio, cgpa=None, 
                          weights={'similarity': 0.4, 'skills': 0.5, 'cgpa': 0.1}):
    """Combines similarity, skill match, and CGPA into a weighted score."""
    skill_score = skill_match_ratio
    cgpa_score = min(cgpa / 4.0, 1.0) if cgpa else 0.5  # Default 0.5 if missing
    return (weights['similarity'] * similarity +
            weights['skills'] * skill_score +
            weights['cgpa'] * cgpa_score) * 100

# Main execution
if __name__ == "__main__":
    # File paths
    job_desc_path = "./job_description.pdf"
    resume_paths = [
        "C:/Users/ASUS/Downloads/Mugesh Rao CV.pdf",
        "C:/Users/ASUS/Desktop/Resume - Elliot.pdf",
        "C:/Users/ASUS/Downloads/Resume Template.pdf"
    ]

    # Extract and preprocess job description
    job_text_raw = extract_text_from_pdf(job_desc_path)
    job_text = preprocess_text(job_text_raw)
    job_skills = set(extract_skills(job_text_raw))
    total_job_skills = max(len(job_skills), 1)  # Avoid division by zero

    # Batch extract and preprocess resumes
    resume_texts_raw = [extract_text_from_pdf(path) for path in resume_paths]
    resume_texts = [preprocess_text(text) for text in resume_texts_raw]

    # Calculate similarities in one pass
    similarities = calculate_similarity(job_text, resume_texts)

    # Process results efficiently
    results = []
    for i, (resume_path, resume_raw, similarity) in enumerate(zip(resume_paths, resume_texts_raw, similarities)):
        cgpa = extract_cgpa(resume_raw)
        email = extract_email(resume_raw)
        resume_skills = set(extract_skills(resume_raw))
        skill_matches = len(job_skills & resume_skills)  # Set intersection
        skill_match_ratio = skill_matches / total_job_skills

        score = compute_weighted_score(similarity, skill_match_ratio, cgpa)

        results.append({
            "Resume": resume_path.split("/")[-1],
            "Match Score (%)": round(score, 2),
            "TF-IDF Similarity (%)": round(similarity * 100, 2),
            "Skill Matches": skill_matches,
            "Matched Skills": ", ".join(resume_skills) or "None",
            "Total Job Skills": total_job_skills,
            "CGPA": cgpa if cgpa else "Not found",
            "Email": email
        })

    # Display and save results
    df = pd.DataFrame(results)
    print("Job Skills Identified:", sorted(job_skills))
    print("\nResume Matching Results:")
    print(df.sort_values("Match Score (%)", ascending=False))
    df.to_csv("resume_matching_results.csv", index=False)
    print("\nResults saved to 'resume_matching_results.csv'")