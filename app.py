# -------------------------------
# app.py
# -------------------------------

import streamlit as st
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
import pdfplumber
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_KEY")
genai.configure(api_key=GEMINI_KEY)

# -------------------------------
# Setup Stopwords
# -------------------------------
nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))
CUSTOM_STOPWORDS = {"resume", "skills", "skill", "experience", "experiences", "worked", "work"}
STOPWORDS = STOPWORDS.union(CUSTOM_STOPWORDS)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    pipeline, label_enc = joblib.load("resume_pipeline_joblib.pkl")
    return pipeline, label_enc

pipeline, le = load_model()

# -------------------------------
# Prediction Function
# -------------------------------
def predict_resume_insights(texts):
    preds = pipeline.predict(texts)
    return le.inverse_transform(preds)

# -------------------------------
# Helper Functions for Insights
# -------------------------------
def extract_skills(texts, top_n=10):
    all_words = []
    for txt in texts:
        words = re.findall(r"\b[a-zA-Z]{3,}\b", str(txt).lower())
        filtered = [w for w in words if w not in STOPWORDS]
        all_words.extend(filtered)
    freq = Counter(all_words)
    return pd.DataFrame(freq.most_common(top_n), columns=["Skill/Keyword", "Count"])

def generate_wordcloud(texts):
    text = " ".join(texts)
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        max_words=200,
        stopwords=STOPWORDS
    ).generate(text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt.gcf())
    plt.clf()

# -------------------------------
# Streamlit App
# -------------------------------
st.title("📊 Veridia Resume Analyzer")
st.write("Upload a single resume (PDF) and get category predictions and AI-powered recommendations.")

# -------------------------------
# Single Resume Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload a PDF Resume", type=["pdf"])
if uploaded_file is not None:
    texts = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
    if not texts:
        st.error("Could not extract text from the PDF.")
    else:
        df = pd.DataFrame({"Resume_str": texts})
        st.write("### Extracted Resume Text (sample)")
        st.dataframe(df.head())

        if st.button("Predict Category"):
            df["Predicted_Category"] = predict_resume_insights(df["Resume_str"].astype(str).tolist())

            # 🔥 BIG Predicted Category
            st.markdown(
                f"""
                <div style='
                    background-color:#f0f8ff;
                    padding:25px;
                    border-radius:10px;
                    text-align:center;
                    margin:20px 0;
                    border:3px solid #0073e6;
                '>
                    <h2 style='color:#0073e6;margin:0;'>✅ Predicted Category</h2>
                    <h1 style='color:#000;font-size:80px;margin:10px 0;font-weight:bold;'>{df['Predicted_Category'][0]}</h1>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Resume Stats
            st.subheader("📄 Resume Stats")
            full_text = " ".join(df["Resume_str"].tolist())
            words = re.findall(r"\b[a-zA-Z]{3,}\b", full_text.lower())
            filtered_words = [w for w in words if w not in STOPWORDS]
            st.markdown(f"- **Total Words:** {len(words)}")
            st.markdown(f"- **Unique Keywords (filtered):** {len(set(filtered_words))}")

            # Word Cloud
            st.subheader("☁️ Key Skills in Resume")
            generate_wordcloud(df["Resume_str"].astype(str).tolist())

            # Top Skills/Keywords (top 10)
            st.subheader("Top 10 Skills/Keywords")
            skills_df = extract_skills(df["Resume_str"].tolist())
            st.dataframe(skills_df)

            # -------------------------------
            # Gemini AI Recommendations
            # -------------------------------
            import google.generativeai as genai

            # Configure the API key
            genai.configure(api_key='AIzaSyDjfigK372IrNsutIcBSLs0bMwMZIuKdzw')

            # Initialize the Gemini model
            model = genai.GenerativeModel('gemini-2.5-flash')

            # Start a chat session
            chat = model.start_chat()

            # Define the prompt for generating recommendations
            prompt = f"""
            You are an HR analytics assistant. The candidate has the following resume content:
            {full_text}

            Predicted category: {df['Predicted_Category'][0]}
            Top skills: {', '.join(skills_df['Skill/Keyword'].tolist())}

            Provide 5 actionable recommendations for hiring strategy, training, and talent management.
            Each recommendation should be concise and 1–2 lines maximum.
            Use bullet points.
            """

            # Send the prompt to the model and get the response
            response = chat.send_message(prompt)

            # Display the AI-generated recommendations
            st.subheader("Recommendations")
            st.markdown(response.text)