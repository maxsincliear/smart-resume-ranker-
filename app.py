import streamlit as st
import fitz  # PyMuPDF
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download only required NLTK components
nltk.download('stopwords')
nltk.download('wordnet')

# --- Text Preprocessing ---
def preprocess_text(text):
    try:
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(text)
        stop_words = set(stopwords.words("english"))
        tokens = [w for w in tokens if w not in stop_words]
        lemmatizer = WordNetLemmatizer()
        lemmatized = [lemmatizer.lemmatize(w) for w in tokens]
        return " ".join(lemmatized)
    except Exception as e:
        st.error(f"Text processing error: {str(e)}")
        return ""

# --- Extract Text from Resume PDF ---
def extract_text_from_pdf(pdf_file):
    try:
        with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

# --- Match Score Calculation ---
def calculate_similarity(resume_text, job_desc):
    try:
        if not resume_text.strip() or not job_desc.strip():
            return 0.0
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform([resume_text, job_desc])
        return round(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100, 2)
    except Exception as e:
        st.error(f"Similarity calculation error: {str(e)}")
        return 0.0

# --- Streamlit Layout ---
st.set_page_config(page_title="Smart Resume Ranker", layout="wide")

# Top Title and Developer Credits
st.markdown("""
    <h1 style='text-align: center; color: #4A90E2;'>Smart Resume Ranker</h1>
    <h4 style='text-align: center; color: #555;'>Built with ❤️ by <strong>Vikas Jaipal</strong> & <strong>Sudha Koushal</strong></h4>
    <hr>
""", unsafe_allow_html=True)

# Layout: Two columns
col1, col2 = st.columns(2)

with col1:
    st.header("📄 Upload Resume")
    resume_file = st.file_uploader("Upload PDF Resume", type=["pdf"])

with col2:
    st.header("📝 Job Description")
    job_title = st.text_input("Job Title")
    job_description = st.text_area("Paste the job description here", height=200)

# Analyze Button
if st.button("🚀 Analyze Match"):
    if resume_file is None or job_description.strip() == "":
        st.warning("⚠️ Please upload a resume and provide the job description.")
    else:
        with st.spinner("Processing resume..."):
            resume_text = extract_text_from_pdf(resume_file)
            preprocessed_resume = preprocess_text(resume_text)
            preprocessed_jd = preprocess_text(job_description)
            match_score = calculate_similarity(preprocessed_resume, preprocessed_jd)

        # Match Score
        st.markdown("---")
        st.subheader("📊 Match Score:")
        st.markdown(f"<h2 style='text-align: center; color: green;'>{match_score}%</h2>", unsafe_allow_html=True)

        if match_score >= 70:
            st.success("✅ Excellent Match: Your resume aligns well with the job!")
        elif match_score >= 40:
            st.info("⚠️ Moderate Match: Try improving your resume.")
        else:
            st.error("❌ Low Match: Your resume doesn't align well. Try tailoring it.")

        # Expanders
        st.markdown("---")
        with st.expander("🔍 View Extracted Resume Text"):
            st.text(resume_text)

        with st.expander("📄 View Job Description"):
            st.text(job_description)
