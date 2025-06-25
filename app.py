import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline
from io import StringIO
from datetime import datetime

# Set page config
st.set_page_config(page_title="AI Feedback Categorizer", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #f4f4f4;
    }
    .reportview-container .markdown-text-container {
        font-family: 'Helvetica';
        color: #333333;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 AI-Generated User Feedback Categorizer")
st.markdown("Automatically classify, analyze, and visualize user reviews for better insights.")

# Load Models
@st.cache_resource(show_spinner=False)
def load_models():
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
    sentiment_analyzer = pipeline("sentiment-analysis", device=-1)
    return classifier, sentiment_analyzer

classifier, sentiment_analyzer = load_models()

# CSV Upload
st.sidebar.header("📁 Upload Feedback CSV")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file with a 'feedback' column", type=["csv"])

if uploaded_file:
    try:
        df_raw = pd.read_csv(uploaded_file)
        if 'review' in df_raw.columns and 'feedback' not in df_raw.columns:
            df_raw.rename(columns={'review': 'feedback'}, inplace=True)
        assert 'feedback' in df_raw.columns, "Missing 'feedback' column."
    except Exception as e:
        st.error(f"Failed to process CSV: {e}")
        st.stop()

    categories = ["Bug", "Feature Request", "Complaint", "Compliment", "General"]

    @st.cache_data(show_spinner=True)
    def process_feedback(_df):
        def classify_and_analyze(text):
            class_result = classifier(text, categories, multi_label=True)
            assigned = [label for label, score in zip(class_result['labels'], class_result['scores']) if score > 0.5] or ["General"]
            sentiment = sentiment_analyzer(text)[0]['label']
            urgency = "High" if any(x in assigned for x in ["Bug", "Complaint"]) else "Medium" if "Feature Request" in assigned else "Low"
            return pd.Series([assigned, sentiment, urgency])

        _df[['categories', 'sentiment', 'urgency']] = _df['feedback'].astype(str).apply(classify_and_analyze)

        # Keywords
        vec = CountVectorizer(stop_words='english', max_features=10)
        keywords = vec.fit_transform(_df['feedback'])
        keyword_list = vec.get_feature_names_out().tolist()
        return _df, keyword_list

    with st.spinner("Processing feedback and generating insights..."):
        df, keywords = process_feedback(df_raw.copy())

    st.success("Feedback processed successfully! ✅")

    # Visual Dashboard
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🗂️ Category Distribution")
        cat_df = pd.DataFrame({'category': [c for cl in df['categories'] for c in cl]})
        fig1, ax1 = plt.subplots()
        sns.countplot(y='category', data=cat_df, ax=ax1, order=cat_df['category'].value_counts().index, palette="viridis")
        ax1.set_title("Number of Feedbacks by Category")
        st.pyplot(fig1)

    with col2:
        st.subheader("💬 Sentiment Distribution")
        fig2, ax2 = plt.subplots()
        sns.countplot(x='sentiment', data=df, palette="Set2", ax=ax2)
        ax2.set_title("Feedback Sentiment")
        st.pyplot(fig2)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("🚨 Urgency Levels")
        fig3, ax3 = plt.subplots()
        sns.countplot(x='urgency', data=df, palette="Reds", ax=ax3)
        ax3.set_title("Feedback Urgency")
        st.pyplot(fig3)

    with col4:
        st.subheader("🔑 Top Keywords")
        st.write(", ".join(keywords))

    st.subheader("📝 Sample Feedback Summaries")
    for idx, row in df.head(10).iterrows():
        st.markdown(f"**{row['feedback']}**")
        st.markdown(f"- Categories: `{', '.join(row['categories'])}`")
        st.markdown(f"- Sentiment: `{row['sentiment']}`, Urgency: `{row['urgency']}`")
        st.markdown("---")

    st.download_button("Download Processed CSV", df.to_csv(index=False), file_name="categorized_feedback.csv")

else:
    st.info("👈 Upload a CSV file with user feedback to begin analysis.")

# Developer Panel
with st.expander("🔧 Developer Notes"):
    st.markdown("""
    - Model: `facebook/bart-large-mnli` for classification, `distilbert-base-uncased-finetuned-sst-2-english` for sentiment.
    - CPU-Only mode enabled for compatibility.
    - Uses multi-label classification and keyword extraction.
    - Designed for feedback categorization from sources like Google Play, Trustpilot, emails.
    """)
