const fs = require('fs');
const pdf = require('pdf-parse');
const { pipeline } = require('@xenova/transformers');

// Load SBERT model
let model;
async function loadModel() {
    model = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    console.log("Model Loaded!");
}

// Extract text from PDF
async function extractTextFromPdf(pdfPath) {
    const dataBuffer = fs.readFileSync(pdfPath);
    const data = await pdf(dataBuffer);
    return data.text;
}

// Compute cosine similarity
function cosineSimilarity(vecA, vecB) {
    let dotProduct = 0, magnitudeA = 0, magnitudeB = 0;
    
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        magnitudeA += vecA[i] ** 2;
        magnitudeB += vecB[i] ** 2;
    }
    
    return dotProduct / (Math.sqrt(magnitudeA) * Math.sqrt(magnitudeB));
}

// Match resume with job description
async function matchResume(jobDescPath, resumePath) {
    try {
        if (!model) await loadModel(); // Load SBERT model if not loaded

        // Extract and encode job description
        const jobText = await extractTextFromPdf(jobDescPath);
        const jobEmbeddingObj = await model(jobText, { pooling: 'mean', normalize: true });

        // Extract and encode resume
        const resumeText = await extractTextFromPdf(resumePath);
        const resumeEmbeddingObj = await model(resumeText, { pooling: 'mean', normalize: true });

        // Extract correct embeddings from returned objects
        const jobEmbedding = jobEmbeddingObj.data;
        const resumeEmbedding = resumeEmbeddingObj.data;

        // Compute similarity score
        const similarity = cosineSimilarity(jobEmbedding, resumeEmbedding) * 100;

        // Output the match score
        console.log(`Match Score for Resume (${resumePath.split('/').pop()}): ${similarity.toFixed(2)}%`);
        return similarity.toFixed(2);
    } catch (error) {
        console.error("Error:", error);
    }
}

// Example Usage
const jobDescPath = "./job_description.pdf";
const resumePath = "C:/Users/ASUS/Downloads/Resume Template.pdf"; // Single resume

matchResume(jobDescPath, resumePath);
