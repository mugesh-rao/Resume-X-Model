import streamlit as st
import fitz  # PyMuPDF
import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import spacy
import matplotlib.pyplot as plt
import seaborn as sns

# Load SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight, fast, and effective

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Extract CGPA with regex
def extract_cgpa(text):
    cgpa_pattern = r'\b(?:CGPA|GPA)\s*:?[\s-]([0-9]+(?:\.[0-9]+)?)\b'
    match = re.search(cgpa_pattern, text, re.IGNORECASE)
    return float(match.group(1)) if match else None

# Extract skills with spaCy
def extract_skills(text, nlp):
    doc = nlp(text)
    skills = {ent.text.lower() for ent in doc.ents if ent.label_ == "SKILL"}
    return skills

# Calculate similarity with SBERT
def calculate_similarity(job_text, resume_text):
    job_embedding = model.encode(job_text, convert_to_tensor=True)
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    similarity = util.cos_sim(job_embedding, resume_embedding).item()
    return similarity

# Weighted scoring
def compute_score(similarity, skill_matches, cgpa, weights={'similarity': 0.4, 'skills': 0.4, 'cgpa': 0.2}):
    skill_score = min(skill_matches / 5, 1.0)  # Normalize (assume 5 skills max)
    cgpa_score = min(cgpa / 4.0, 1.0) if cgpa else 0.5  # Normalize (assume 4.0 scale)
    total_score = (weights['similarity'] * similarity +
                   weights['skills'] * skill_score +
                   weights['cgpa'] * cgpa_score)
    return total_score * 100  # Scale to 0-100

# Streamlit App
st.title("Resume Matching Tool with SBERT")

# File Upload
st.sidebar.header("Upload Files")
resumes = st.sidebar.file_uploader("Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)
job_desc = st.sidebar.file_uploader("Upload Job Description (PDF)", type="pdf")

if resumes and job_desc:
    # Load spaCy model (assuming a custom model for skills)
    nlp = spacy.load("en_core_web_sm")  # Replace with your custom model if available

    # Extract job description text
    job_text = extract_text_from_pdf(job_desc)
    job_skills = extract_skills(job_text, nlp)

    # Process resumes
    results = []
    for resume in resumes:
        resume_text = extract_text_from_pdf(resume)
        cgpa = extract_cgpa(resume_text)
        resume_skills = extract_skills(resume_text, nlp)
        skill_matches = len(job_skills.intersection(resume_skills))
        similarity = calculate_similarity(job_text, resume_text)
        score = compute_score(similarity, skill_matches, cgpa)

        results.append({
            "Resume": resume.name,
            "Score": score,
            "Similarity": similarity,
            "Skill Matches": skill_matches,
            "CGPA": cgpa
        })

    # Display results
    df = pd.DataFrame(results)
    st.subheader("Matching Results")
    st.dataframe(df.sort_values("Score", ascending=False))

    # Heatmap of similarity
    st.subheader("Similarity Heatmap")
    resume_names = [r["Resume"] for r in results]
    similarity_scores = [r["Similarity"] for r in results]
    heatmap_data = pd.DataFrame([similarity_scores], columns=resume_names, index=["Job Description"])
    
    fig, ax = plt.subplots()
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
    st.pyplot(fig)

else:
    st.warning("Please upload both resumes and a job description.")