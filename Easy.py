import fitz  # PyMuPDF
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


nltk.download('punkt_tab')
# Manually download necessary packages
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')  # For WordNet to work properly
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Predefined list of skills (expand this based on your domain)
SKILLS_LIST = [
    "data analysis", "data visualization", "data mining", "data cleaning", "data wrangling",
    "python", "r", "sql", "excel", "tableau", "power bi", "machine learning", 
    "statistical analysis", "predictive modeling", "big data", "aws", "cloud computing",
    "docker", "git", "communication", "project management", "linux"
]

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to preprocess text
def preprocess_text(text):
    """Cleans and tokenizes text."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Function to extract CGPA
def extract_cgpa(text):
    """Extracts CGPA using regex."""
    cgpa_pattern = r'\b(?:CGPA|GPA)\s*:?[\s-]([0-9]+(?:\.[0-9]+)?)\b'
    match = re.search(cgpa_pattern, text, re.IGNORECASE)
    return float(match.group(1)) if match else None

# Function to extract email
def extract_email(text):
    """Extracts email using regex."""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    match = re.search(email_pattern, text, re.IGNORECASE)
    return match.group(0) if match else "Not found"

# Function to extract skills
def extract_skills(text, skills_list=SKILLS_LIST):
    """Extracts skills from text based on a predefined list."""
    text = text.lower()
    found_skills = [skill for skill in skills_list if re.search(r'\b' + re.escape(skill) + r'\b', text)]
    return found_skills

# Function to calculate TF-IDF similarity
def calculate_similarity(job_text, resume_texts):
    """Calculates TF-IDF cosine similarity between job description and resumes."""
    all_texts = [job_text] + resume_texts
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)  # Increased for more granularity
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    job_vector = tfidf_matrix[0]
    resume_vectors = tfidf_matrix[1:]
    similarities = cosine_similarity(job_vector, resume_vectors).flatten()
    return similarities

# Function to compute weighted score
def compute_weighted_score(similarity, skill_match_ratio, cgpa=None, 
                          weights={'similarity': 0.4, 'skills': 0.5, 'cgpa': 0.1}):
    """Combines similarity, skill match, and CGPA into a weighted score."""
    skill_score = skill_match_ratio  # Already a ratio (0-1)
    cgpa_score = min(cgpa / 4.0, 1.0) if cgpa else 0.5  # Normalize CGPA (assume max 4.0)
    total_score = (weights['similarity'] * similarity +
                   weights['skills'] * skill_score +
                   weights['cgpa'] * cgpa_score)
    return total_score * 100  # Scale to 0-100

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
    job_skills = extract_skills(job_text_raw)
    total_job_skills = len(job_skills) if job_skills else 1  # Avoid division by zero

    # Extract and preprocess resumes
    resume_texts_raw = [extract_text_from_pdf(path) for path in resume_paths]
    resume_texts = [preprocess_text(text) for text in resume_texts_raw]

    # Calculate TF-IDF similarities
    similarities = calculate_similarity(job_text, resume_texts)

    # Process results
    results = []
    for i, resume_path in enumerate(resume_paths):
        # Extract features
        cgpa = extract_cgpa(resume_texts_raw[i])
        email = extract_email(resume_texts_raw[i])
        resume_skills = extract_skills(resume_texts_raw[i])
        skill_matches = len(set(job_skills).intersection(resume_skills))
        skill_match_ratio = skill_matches / total_job_skills if total_job_skills > 0 else 0

        # Compute weighted score
        score = compute_weighted_score(similarities[i], skill_match_ratio, cgpa)

        # Store results
        results.append({
            "Resume": resume_path.split("/")[-1],
            "Match Score (%)": round(score, 2),
            "TF-IDF Similarity (%)": round(similarities[i] * 100, 2),
            "Skill Matches": skill_matches,
            "Matched Skills": ", ".join(resume_skills) if resume_skills else "None",
            "Total Job Skills": total_job_skills,
            "CGPA": cgpa if cgpa else "Not found",
            "Email": email
        })

    # Display results
    df = pd.DataFrame(results)
    print("Job Skills Identified:", job_skills)
    print("\nResume Matching Results:")
    print(df.sort_values("Match Score (%)", ascending=False))

    # Optional: Save to CSV
    df.to_csv("resume_matching_results.csv", index=False)
    print("\nResults saved to 'resume_matching_results.csv'")