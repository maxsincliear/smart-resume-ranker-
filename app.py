import streamlit as st
import fitz  # PyMuPDF
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

# Download only needed NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# ------------------ Preprocessing ------------------
def read_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [w for w in tokens if w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(lemmatized)

def calculate_similarity(resume_text, job_desc):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([resume_text, job_desc])
    return round(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100, 2)

# ------------------ Streamlit App ------------------
st.set_page_config(page_title="Smart Resume Ranker", layout="wide")

st.markdown("<h1 style='text-align: center; color: white; background-color: #2a5298; padding: 1rem; border-radius: 10px;'>🧠 Smart Resume Ranker & Analysis</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3178/3178374.png", width=80)
    st.markdown("## 🔍 Navigation")
    st.markdown("✅ Upload Resume(s)")
    st.markdown("✅ Enter Job Description")
    st.markdown("✅ View Matching Results")
    st.markdown("✅ Insights & Graphs")
    st.markdown("---")
    st.markdown("### 👨‍💻 Developed by:")
    st.markdown("**Vikas Jaipal (22EEBIT009)**")
    st.markdown("**Sudha Koushal (22EEBIT006)**")

st.subheader("Welcome to Smart Resume Ranker & Analysis 🚀")
st.markdown("Upload resumes, enter a job description, and get instant AI-based matching and analysis!")

st.markdown("### 📂 Step 1: Upload Resume PDFs")
uploaded_files = st.file_uploader("Upload one or more resumes", type="pdf", accept_multiple_files=True)

st.markdown("### 📝 Step 2: Enter Job Description")
job_desc = st.text_area("Paste the job description here", height=200)

if st.button("🔍 Analyze & Match"):
    if not uploaded_files or not job_desc.strip():
        st.warning("Please upload at least one resume and enter a job description.")
    else:
        st.success("Processing... Please wait ⏳")
        job_clean = preprocess_text(job_desc)

        for file in uploaded_files:
            text = read_pdf(file)
            resume_clean = preprocess_text(text)
            score = calculate_similarity(resume_clean, job_clean)

            st.markdown(f"### 📄 {file.name}")
            st.write(f"**Match Score:** `{score}%`")
            if score >= 70:
                st.success("✅ Excellent match!")
            elif score >= 50:
                st.info("⚠️ Moderate match")
            else:
                st.warning("❌ Low match")

st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 14px;'>Project developed by <b>Vikas Jaipal</b> & <b>Sudha Koushal</b> — 3rd Year IT | Engineering College Bikaner</p>", unsafe_allow_html=True)
