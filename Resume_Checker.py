import streamlit as st
import io
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
import spacy
import pdfplumber
import docx
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.cli import download

# Download necessary NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load spaCy model
@st.cache_resource
def load_nlp_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_nlp_model()

# --- HELPER FUNCTIONS ---
def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file."""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages if page.extract_text())
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_docx(docx_file):
    """Extracts text from a DOCX file."""
    try:
        doc = docx.Document(docx_file)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {e}")
        return ""

def extract_text_from_txt(txt_file):
    """Extracts text from a plain text file."""
    try:
        text_content = txt_file.read()
        if isinstance(text_content, bytes):
            return text_content.decode('utf-8')
        return text_content
    except Exception as e:
        st.error(f"Error extracting text from TXT: {e}")
        return ""

def preprocess_text(text):
    """Preprocesses the text for analysis (lowercasing, removing punctuation, stop words)."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [w for w in word_tokens if w not in stop_words]
    return " ".join(filtered_text)

def calculate_keyword_similarity(resume_text, job_description_text):
    """Calculates the keyword similarity between the resume and job description."""
    if not resume_text or not job_description_text:
        return 0.0
    vectorizer = TfidfVectorizer()
    try:
        vectors = vectorizer.fit_transform([resume_text, job_description_text])
        similarity = (vectors * vectors.T).toarray()[0,1]
        return similarity
    except Exception as e:
        st.warning(f"Error calculating similarity: {e}")
        return 0.0

def analyze_experience(resume_text):
    """Analyzes the experience section for keywords and completeness."""
    if not resume_text:
        return "No resume text to analyze. ❌"
    if "experience" in resume_text.lower():
        return "Experience section found. ✅"
    else:
        return "Experience section not found. Consider adding it. ❌"

def analyze_skills(resume_text, job_description_text):
    """Analyzes the skills section and compares against required skills in job description."""
    if not resume_text or not job_description_text:
        return set(), set()
    
    resume_doc = nlp(resume_text)
    jd_doc = nlp(job_description_text)
    
    resume_skills = [token.text for token in resume_doc if token.pos_ in ("NOUN", "ADJ")]
    jd_skills = [token.text for token in jd_doc if token.pos_ in ("NOUN", "ADJ")]

    common_skills = set(resume_skills).intersection(set(jd_skills))
    missing_skills = set(jd_skills) - set(resume_skills)

    # Limit to meaningful skills (more than 3 characters)
    common_skills = {skill for skill in common_skills if len(skill) > 3}
    missing_skills = {skill for skill in missing_skills if len(skill) > 3}

    return common_skills, missing_skills

def analyze_education(resume_text):
    """Analyzes the education section for completeness."""
    if not resume_text:
        return "No resume text to analyze. ❌"
    if "education" in resume_text.lower() or "degree" in resume_text.lower():
        return "Education section found. ✅"
    else:
        return "Education section not found. Consider adding it. ❌"

def analyze_achievements(resume_text):
    """Analyzes the achievements section for quantifiable results."""
    if not resume_text:
        return "No resume text to analyze. ❌"
    if re.search(r'\d+', resume_text):
        return "Achievements with quantifiable results found. ✅"
    else:
        return "Consider adding achievements with quantifiable results. ❌"

def assess_formatting(resume_text):
    """Assess resume for formatting issues."""
    if not resume_text:
        return "No resume text to analyze. ❌"
    
    has_clear_headings = any(heading in resume_text.lower() for heading in ["education", "experience", "skills", "summary"])
    if not has_clear_headings:
        return "Resume lacks clear section headings. Use headings like 'Experience', 'Education', 'Skills'. ❌"

    if not re.search(r"•|\-|\*", resume_text):
        return "Experience section should use bullet points to highlight achievements. ❌"

    return "Formatting appears generally good. ✅"

def generate_improved_resume(resume_text, missing_skills):
    """Generates a placeholder improved resume with suggestions."""
    if not resume_text:
        return "Please upload a resume to get improvement suggestions."
    
    improved_resume = resume_text

    if missing_skills:
        improved_resume += "\n\n**Recommendation:** Consider adding the following skills: " + ", ".join(missing_skills)

    return improved_resume

def calculate_overall_score(similarity_score, experience_feedback, common_skills, missing_skills, education_feedback, achievement_feedback, formatting_feedback):
    """Calculates an overall resume score based on individual metrics."""
    score = 0

    # Weighting factors
    similarity_weight = 0.25
    experience_weight = 0.15
    skills_weight = 0.20
    education_weight = 0.10
    achievement_weight = 0.10
    formatting_weight = 0.10

    # Similarity Score
    score += similarity_score * similarity_weight

    # Experience Score
    if "✅" in experience_feedback:
        score += 1 * experience_weight

    # Skills Score
    total_skills = len(missing_skills) + len(common_skills)
    if total_skills > 0:
        score += (len(common_skills) / total_skills) * skills_weight
    
    # Education Score
    if "✅" in education_feedback:
        score += 1 * education_weight

    # Achievement Score
    if "✅" in achievement_feedback:
        score += 1 * achievement_weight

    # Formatting Score
    if "✅" in formatting_feedback:
        score += 1 * formatting_weight

    return score * 100  # Scale to 100

# --- STREAMLIT APP ---
def main():
    st.title("Resume Analyzer 📄")

    # Sidebar for settings
    with st.sidebar:
        st.header("Settings ⚙️")

    # File Upload
    uploaded_file = st.file_uploader("Upload your resume (PDF, DOCX, TXT) 📂", type=["pdf", "docx", "txt"])

    # Job Description Input
    job_description = st.text_area("Paste the job description here: 📝")

    # Proceed button
    proceed_button = st.button("Proceed with Analysis 🔍")

    if proceed_button and uploaded_file and job_description:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        resume_text = ""

        with st.spinner("Extracting text from resume... ⏳"):
            if file_extension == "pdf":
                resume_text = extract_text_from_pdf(uploaded_file)
            elif file_extension == "docx":
                resume_text = extract_text_from_docx(uploaded_file)
            elif file_extension == "txt":
                resume_text = extract_text_from_txt(uploaded_file)
            else:
                st.error("Unsupported file format. ❌")
                return

        if resume_text:
            with st.spinner("Analyzing resume... ⏳"):
                # Preprocess Text
                processed_resume = preprocess_text(resume_text)
                processed_jd = preprocess_text(job_description)

                # Calculate Similarity
                similarity_score = calculate_keyword_similarity(processed_resume, processed_jd)

                # Experience Analysis
                experience_feedback = analyze_experience(resume_text)

                # Skills Analysis
                common_skills, missing_skills = analyze_skills(resume_text, job_description)

                # Education Analysis
                education_feedback = analyze_education(resume_text)

                # Achievement Analysis
                achievement_feedback = analyze_achievements(resume_text)

                # Formatting Assessment
                formatting_feedback = assess_formatting(resume_text)

                # Calculate Overall Score
                overall_score = calculate_overall_score(
                    similarity_score, 
                    experience_feedback, 
                    common_skills, 
                    missing_skills, 
                    education_feedback, 
                    achievement_feedback, 
                    formatting_feedback
                )
                
                # Display results in a nice format
                st.subheader("Analysis Results")
                st.metric(label="Overall Resume Score", value=f"{overall_score:.2f}/100 🏆")
                st.write("This score reflects the overall quality and completeness of your resume based on the factors analyzed.")
                
                # Create two columns for the results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Match Score:** {similarity_score:.2f} 🔍")
                    st.write(f"**Experience Analysis:** {experience_feedback}")
                    st.write(f"**Education Analysis:** {education_feedback}")
                
                with col2:
                    st.write(f"**Achievement Analysis:** {achievement_feedback}")
                    st.write(f"**Formatting Assessment:** {formatting_feedback}")
                
                # Skills section
                st.subheader("Skills Analysis")
                
                common_skills_list = list(common_skills)
                if common_skills_list:
                    st.write("**Common Skills:**")
                    for i in range(0, len(common_skills_list), 3):
                        cols = st.columns(3)
                        for j in range(3):
                            if i+j < len(common_skills_list):
                                cols[j].write(f"✅ {common_skills_list[i+j]}")
                else:
                    st.write("**No common skills found.**")
                
                missing_skills_list = list(missing_skills)
                if missing_skills_list:
                    st.write("**Missing Skills:**")
                    for i in range(0, len(missing_skills_list), 3):
                        cols = st.columns(3)
                        for j in range(3):
                            if i+j < len(missing_skills_list):
                                cols[j].write(f"❌ {missing_skills_list[i+j]}")
                else:
                    st.write("**No missing skills identified.**")

                # Generate Improved Resume
                improved_resume = generate_improved_resume(resume_text, missing_skills)
                
                st.subheader("Improved Resume Suggestions ✨")
                st.text_area("Resume with Suggestions", improved_resume, height=300)

                # Download Improved Resume
                st.download_button(
                    label="Download Improved Resume (TXT) 📥",
                    data=improved_resume.encode("utf-8"),
                    file_name="improved_resume.txt",
                    mime="text/plain",
                )
        else:
            st.error("Could not extract text from the uploaded file. Please try another file.")
    elif proceed_button:
        if not uploaded_file:
            st.warning("Please upload your resume to proceed.")
        if not job_description:
            st.warning("Please paste the job description to proceed.")

if __name__ == "__main__":
    main()