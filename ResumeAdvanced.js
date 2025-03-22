const fs = require('fs');
const pdf = require('pdf-parse');
const natural = require('natural');
const { pipeline } = require('@xenova/transformers');
const stopwords = require('stopwords').english;

// Predefined list of skills (domain-specific)
const SKILLS_LIST = new Set([
    "data analysis", "data visualization", "data mining", "data cleaning", "data wrangling",
    "python", "r", "sql", "excel", "tableau", "power bi", "machine learning",
    "statistical analysis", "predictive modeling", "big data", "aws", "cloud computing",
    "docker", "git", "communication", "project management", "linux"
]);

// Regex patterns
const CGPA_PATTERN = /\b(?:CGPA|GPA)\s*:?[\s-]([0-9]+(?:\.[0-9]+)?)\b/i;
const EMAIL_PATTERN = /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/i;
const EXPERIENCE_PATTERN = /(\d+)[\s+](?:years?|yrs?)/gi;

// Cache for text processing
const textCache = new Map();
let sentenceEmbedder = null;

// Initialize the sentence embedder
async function initializeEmbedder() {
    if (!sentenceEmbedder) {
        sentenceEmbedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    }
    return sentenceEmbedder;
}

// Extract text from PDF with caching
async function extractTextFromPdf(pdfPath) {
    if (textCache.has(pdfPath)) {
        return textCache.get(pdfPath);
    }

    try {
        const dataBuffer = fs.readFileSync(pdfPath);
        const data = await pdf(dataBuffer);
        const text = data.text;
        textCache.set(pdfPath, text);
        return text;
    } catch (error) {
        console.error(`Error extracting text from PDF at ${pdfPath}:`, error);
        throw error; // Rethrow to handle in the calling function
    }
}

// Text preprocessing
function preprocessText(text) {
    const tokenizer = new natural.WordTokenizer();
    return tokenizer.tokenize(text.toLowerCase())
        .filter(word => !stopwords.includes(word))
        .join(' ');
}

// Get text embeddings
async function getEmbedding(text) {
    const embedder = await initializeEmbedder();
    const result = await embedder(text, { pooling: 'mean', normalize: true });
    return result.data;
}

// Calculate cosine similarity
function cosineSimilarity(vec1, vec2) {
    const dotProduct = vec1.reduce((sum, val, i) => sum + val * vec2[i], 0);
    const norm1 = Math.sqrt(vec1.reduce((sum, val) => sum + val * val, 0));
    const norm2 = Math.sqrt(vec2.reduce((sum, val) => sum + val * val, 0));
    return dotProduct / (norm1 * norm2);
}

// Extract skills with context
function extractSkills(text, context = 5) {
    const words = text.toLowerCase().split(/\s+/);
    const skills = new Set();
    
    SKILLS_LIST.forEach(skill => {
        if (text.toLowerCase().includes(skill)) {
            skills.add(skill);
        }
    });
    
    return Array.from(skills);
}

// Extract CGPA
function extractCgpa(text) {
    const match = text.match(CGPA_PATTERN);
    return match ? parseFloat(match[1]) : null;
}

// Extract email
function extractEmail(text) {
    const match = text.match(EMAIL_PATTERN);
    return match ? match[0] : "Not found";
}

// Compute final score
function computeScore(semanticSimilarity, skillMatchRatio, cgpa, experience) {
    const weights = {
        semantic: 0.35,
        skills: 0.35,
        cgpa: 0.15,
        experience: 0.15
    };

    const cgpaScore = cgpa ? Math.min(cgpa / 4.0, 1.0) : 0.5;
    const experienceScore = Math.min(experience / 10, 1);

    return (
        weights.semantic * semanticSimilarity +
        weights.skills * skillMatchRatio +
        weights.cgpa * cgpaScore +
        weights.experience * experienceScore
    ) * 100;
}

// Main execution function
async function matchResumes(jobDescPath, resumePaths) {
    try {
        // Process job description
        const jobTextRaw = await extractTextFromPdf(jobDescPath);
        const jobText = preprocessText(jobTextRaw);
        const jobEmbedding = await getEmbedding(jobText);
        const jobSkills = new Set(extractSkills(jobTextRaw));
        const totalJobSkills = Math.max(jobSkills.size, 1);

        // Process resumes
        const results = await Promise.all(resumePaths.map(async (resumePath) => {
            try {
                const resumeTextRaw = await extractTextFromPdf(resumePath);
                const resumeText = preprocessText(resumeTextRaw);
                const resumeEmbedding = await getEmbedding(resumeText);

                // Calculate similarity
                const similarity = cosineSimilarity(jobEmbedding, resumeEmbedding);

                // Extract features
                const cgpa = extractCgpa(resumeTextRaw);
                const email = extractEmail(resumeTextRaw);
                const resumeSkills = new Set(extractSkills(resumeTextRaw));
                const skillMatches = new Set([...jobSkills].filter(x => resumeSkills.has(x))).size;
                const skillMatchRatio = skillMatches / totalJobSkills;

                // Extract experience
                const experienceYears = extractExperience(resumeTextRaw);

                // Calculate final score
                const score = computeScore(similarity, skillMatchRatio, cgpa, experienceYears);

                return {
                    "Resume": resumePath.split("/").pop().slice(0, 5),
                    "Match Score (%)": score.toFixed(2),
                    "Semantic Similarity (%)": (similarity * 100).toFixed(2),
                    "CGPA": cgpa || "Not found",
                    "Email": email
                };
            } catch (error) {
                console.error(`Error processing resume at ${resumePath}:`, error);
                return null; // Return null for this resume to skip it
            }
        }));

        // Filter out null results
        const filteredResults = results.filter(result => result !== null);

        // Sort results
        filteredResults.sort((a, b) => b["Match Score (%)"] - a["Match Score (%)"]);
        
        // Display results
        console.table(filteredResults);

        // Save to CSV
        const csv = [
            Object.keys(filteredResults[0]).join(','),
            ...filteredResults.map(row => Object.values(row).join(','))
        ].join('\n');

        fs.writeFileSync('resume_matching_results.csv', csv);
        console.log("\nResults saved to 'resume_matching_results.csv'");

    } catch (error) {
        console.error("Error in matchResumes function:", error);
    }
}

// Helper function to extract experience
function extractExperience(text) {
    const matches = [...text.matchAll(EXPERIENCE_PATTERN)];
    if (matches.length > 0) {
        return Math.max(...matches.map(m => parseInt(m[1])));
    }
    return 0;
}

// Example usage
const jobDescPath = "./job_description.pdf";
const resumePaths = [
    "C:/Users/ASUS/Downloads/Mugesh Rao CV.pdf",
    "C:/Users/ASUS/Desktop/Resume - Elliot.pdf",
    "C:/Users/ASUS/Downloads/Resume Template.pdf"
];

matchResumes(jobDescPath, resumePaths); 