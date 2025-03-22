const fs = require('fs');
const pdf = require('pdf-parse');
const { pipeline } = require('@xenova/sentence-transformers');

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
    const dotProduct = vecA.reduce((sum, val, i) => sum + val * vecB[i], 0);
    const magnitudeA = Math.sqrt(vecA.reduce((sum, val) => sum + val ** 2, 0));
    const magnitudeB = Math.sqrt(vecB.reduce((sum, val) => sum + val ** 2, 0));
    return dotProduct / (magnitudeA * magnitudeB);
}

// Match resume with job description
async function matchResume(jobDescPath, resumePath) {
    try {
        if (!model) await loadModel(); // Load SBERT model if not loaded

        // Extract and encode job description
        const jobText = await extractTextFromPdf(jobDescPath);
        const jobEmbedding = await model(jobText, { pooling: 'mean', normalize: true });

        // Extract and encode resume
        const resumeText = await extractTextFromPdf(resumePath);
        const resumeEmbedding = await model(resumeText, { pooling: 'mean', normalize: true });

        // Compute similarity score
        const similarity = cosineSimilarity(jobEmbedding, resumeEmbedding) * 100;

        // Output the match score
        console.log(`Match Score for Resume (${resumePath}): ${similarity.toFixed(2)}%`);
        return similarity.toFixed(2);
    } catch (error) {
        console.error("Error:", error);
    }
}

// Example Usage
const jobDescPath = "./job_description.pdf";
const resumePath = "C:/Users/ASUS/Downloads/Mugesh Rao CV.pdf"; // Single resume

matchResume(jobDescPath, resumePath);
