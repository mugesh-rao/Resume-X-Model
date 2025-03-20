const fs = require('fs');
const pdf = require('pdf-parse');
const natural = require('natural');
const { TfIdf } = natural;
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

// Cache for PDF text extraction
const textCache = new Map();

// Extract text from PDF
async function extractTextFromPdf(pdfPath) {
    if (textCache.has(pdfPath)) {
        return textCache.get(pdfPath);
    }

    const dataBuffer = fs.readFileSync(pdfPath);
    const data = await pdf(dataBuffer);
    const text = data.text;
    textCache.set(pdfPath, text);
    return text;
}

// Preprocess text
function preprocessText(text) {
    const tokens = text.toLowerCase()
        .replace(/[^\w\s]/g, '')
        .split(/\s+/)
        .filter(word => !stopwords.includes(word));
    return tokens.join(' ');
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

// Extract skills
function extractSkills(text) {
    const textLower = text.toLowerCase();
    return Array.from(SKILLS_LIST).filter(skill => textLower.includes(skill));
}

// Calculate TF-IDF similarity
function calculateSimilarity(jobText, resumeTexts) {
    const tfidf = new TfIdf();
    
    // Add documents
    tfidf.addDocument(jobText);
    resumeTexts.forEach(text => tfidf.addDocument(text));

    // Calculate similarities
    return resumeTexts.map((_, index) => {
        let similarity = 0;
        let totalWeight = 0;

        tfidf.listTerms(0).forEach(item => {
            const jobWeight = item.tfidf;
            const resumeWeight = tfidf.tfidf(item.term, index + 1);
            similarity += jobWeight * resumeWeight;
            totalWeight += jobWeight * jobWeight;
        });

        return similarity / Math.sqrt(totalWeight);
    });
}

// Compute weighted score
function computeWeightedScore(similarity, skillMatchRatio, cgpa, weights = {
    similarity: 0.4,
    skills: 0.5,
    cgpa: 0.1
}) {
    const cgpaScore = cgpa ? Math.min(cgpa / 4.0, 1.0) : 0.5;
    return (weights.similarity * similarity +
            weights.skills * skillMatchRatio +
            weights.cgpa * cgpaScore) * 100;
}

// Main execution function
async function matchResumes(jobDescPath, resumePaths) {
    try {
        // Extract and preprocess job description
        const jobTextRaw = await extractTextFromPdf(jobDescPath);
        const jobText = preprocessText(jobTextRaw);
        const jobSkills = new Set(extractSkills(jobTextRaw));
        const totalJobSkills = Math.max(jobSkills.size, 1);

        // Extract and preprocess resumes
        const resumeTextsRaw = await Promise.all(
            resumePaths.map(path => extractTextFromPdf(path))
        );
        const resumeTexts = resumeTextsRaw.map(preprocessText);

        // Calculate similarities
        const similarities = calculateSimilarity(jobText, resumeTexts);

        // Process results
        const results = resumePaths.map((resumePath, i) => {
            const cgpa = extractCgpa(resumeTextsRaw[i]);
            const email = extractEmail(resumeTextsRaw[i]);
            const resumeSkills = new Set(extractSkills(resumeTextsRaw[i]));
            const skillMatches = new Set([...jobSkills].filter(x => resumeSkills.has(x))).size;
            const skillMatchRatio = skillMatches / totalJobSkills;
            const score = computeWeightedScore(similarities[i], skillMatchRatio, cgpa);

            return {
                "Resume": resumePath.split("/").pop(),
                "Match Score (%)": score.toFixed(2),
                "TF-IDF Similarity (%)": (similarities[i] * 100).toFixed(2),
                "Skill Matches": skillMatches,
                "Matched Skills": Array.from(resumeSkills).join(", ") || "None",
                "Total Job Skills": totalJobSkills,
                "CGPA": cgpa || "Not found",
                "Email": email
            };
        });

        // Sort results by match score
        results.sort((a, b) => b["Match Score (%)"] - a["Match Score (%)"]);

        // Output results
        console.log("Job Skills Identified:", Array.from(jobSkills).sort());
        console.log("\nResume Matching Results:");
        console.table(results);

        // Save to CSV
        const csv = [
            Object.keys(results[0]).join(','),
            ...results.map(row => Object.values(row).join(','))
        ].join('\n');

        fs.writeFileSync('resume_matching_results.csv', csv);
        console.log("\nResults saved to 'resume_matching_results.csv'");

    } catch (error) {
        console.error("Error:", error);
    }
}

// Example usage
const jobDescPath = "./job_description.pdf";
const resumePaths = [
    "C:/Users/ASUS/Downloads/Mugesh Rao CV.pdf",
    "C:/Users/ASUS/Desktop/Resume - Elliot.pdf",
    "C:/Users/ASUS/Downloads/Resume Template.pdf"
];

matchResumes(jobDescPath, resumePaths); 