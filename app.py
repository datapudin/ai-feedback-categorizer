import ast
from pathlib import Path

# Redefine app_code after code execution state reset
app_code = """
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime
import io

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="AI Feedback Categorizer", layout="wide")
st.title("🧠 AI-Generated User Feedback Categorizer")

# -------------------- MODEL LOADING (CACHE) --------------------
@st.cache_resource
def load_models():
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        device=-1
    )
    classifier_model = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1
    )
    return classifier_model, sentiment_model

classifier, sentiment_analyzer = load_models()

# -------------------- FILE UPLOAD --------------------
st.sidebar.header("📂 Upload Feedback CSV")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file with feedback column", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Rename if column is 'review'
        if 'feedback' not in df.columns:
            if 'review' in df.columns:
                df.rename(columns={'review': 'feedback'}, inplace=True)
            else:
                st.error("⚠️ CSV must contain a 'feedback' or 'review' column.")
                st.stop()

        df.dropna(subset=['feedback'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        st.success(f"✅ Uploaded {len(df)} rows.")
        st.write(df.head())

        # -------------------- ANALYSIS --------------------
        st.subheader("🔍 Processing Feedback...")

        categories = ["Bug", "Feature Request", "Complaint", "Compliment", "General"]

        def process_feedback(text):
            class_result = classifier(text, candidate_labels=categories, multi_label=True)
            assigned = [label for label, score in zip(class_result['labels'], class_result['scores']) if score > 0.5]
            if not assigned:
                assigned = ["General"]
            sentiment = sentiment_analyzer(text)[0]['label']
            if "Bug" in assigned or "Complaint" in assigned:
                urgency = "High"
            elif "Feature Request" in assigned:
                urgency = "Medium"
            else:
                urgency = "Low"
            return pd.Series([assigned, sentiment, urgency])

        with st.spinner("Running LLM-based classification..."):
            df[['categories', 'sentiment', 'urgency']] = df['feedback'].apply(process_feedback)
            vectorizer = CountVectorizer(stop_words='english', max_features=10)
            X = vectorizer.fit_transform(df['feedback'])
            keywords = vectorizer.get_feature_names_out()

        # -------------------- VISUALIZATIONS --------------------
        st.subheader("📊 Category Distribution")
        all_cats = [c for sublist in df['categories'] for c in sublist]
        cat_df = pd.DataFrame({'category': all_cats})
        st.bar_chart(cat_df['category'].value_counts())

        st.subheader("📊 Sentiment Distribution")
        st.bar_chart(df['sentiment'].value_counts())

        st.subheader("📊 Urgency Distribution")
        st.bar_chart(df['urgency'].value_counts())

        st.subheader("🗝️ Top Keywords")
        st.markdown(", ".join(keywords))

        st.subheader("🔎 Search Feedback")
        search_term = st.text_input("Enter keyword to search in feedback:")
        if search_term:
            filtered = df[df['feedback'].str.contains(search_term, case=False)]
            st.write(f"Found {len(filtered)} results:")
            st.dataframe(filtered[['feedback', 'categories', 'sentiment', 'urgency']])

        st.subheader("📋 Feedback Summaries")
        for i, row in df.iterrows():
            st.markdown(f"**{i+1}. {row['feedback']}**")
            st.markdown(f"- Categories: `{row['categories']}`")
            st.markdown(f"- Sentiment: `{row['sentiment']}`, Urgency: `{row['urgency']}`")

        # -------------------- REPORT SUMMARY --------------------
        st.subheader("📑 Summary Report")
        total = len(df)
        pos = (df['sentiment'] == 'POSITIVE').sum()
        neg = (df['sentiment'] == 'NEGATIVE').sum()
        high_urgency = (df['urgency'] == 'High').sum()

        st.markdown(f"**Total Feedback Analyzed:** {total}")
        st.markdown(f"**Positive Feedback:** {pos}")
        st.markdown(f"**Negative Feedback:** {neg}")
        st.markdown(f"**High Urgency Issues:** {high_urgency}")

        # -------------------- CSV DOWNLOAD --------------------
        st.subheader("📥 Download Results")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download categorized data as CSV",
            data=csv_buffer.getvalue(),
            file_name=f"categorized_feedback_{timestamp}.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"🚫 Error processing file: {e}")

else:
    st.info("👈 Upload a CSV file with a 'feedback' or 'review' column to begin.")
"""

# Check for syntax errors
try:
    ast.parse(app_code)
    result = "✅ No syntax errors detected in app.py."
except SyntaxError as e:
    result = f"❌ Syntax error in app.py: {e}"

result
