import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import io
import warnings
from transformers import pipeline

warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(page_title="AI Feedback Categorizer", layout="wide")
st.title("🤖 AI-Generated User Feedback Categorizer")

st.sidebar.header("Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'review' in df.columns:
        df.rename(columns={'review': 'feedback'}, inplace=True)

    st.write("### Sample Data")
    st.dataframe(df.head())

    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)

    categories = ["Bug Report", "Feature Request", "Complaint", "Compliment", "General Feedback"]

    st.write("### Processing Feedback...")
    feedback_texts = df['feedback'].astype(str).tolist()

    results = []
    for feedback in feedback_texts:
        try:
            classification = classifier(feedback, categories)
            top_category = classification['labels'][0]
            sentiment = sentiment_analyzer(feedback)[0]
            urgency = "High" if sentiment['label'] == "NEGATIVE" and sentiment['score'] > 0.8 else "Low"

            results.append({
                "feedback": feedback,
                "category": top_category,
                "sentiment": sentiment['label'],
                "sentiment_score": round(sentiment['score'], 2),
                "urgency": urgency
            })
        except Exception as e:
            results.append({"feedback": feedback, "category": "Error", "sentiment": "Error", "sentiment_score": 0, "urgency": "Low"})

    df_result = pd.DataFrame(results)
    st.write("### Categorized Feedback")
    st.dataframe(df_result.head())

    st.write("### Insights Dashboard")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    sns.countplot(y='category', hue='category', data=df_result, ax=ax1,
                  order=df_result['category'].value_counts().index, palette="viridis", legend=False)
    ax1.set_title("Category Distribution")

    sns.countplot(x='sentiment', hue='sentiment', data=df_result, ax=ax2,
                  palette="Set2", legend=False)
    ax2.set_title("Sentiment Distribution")

    sns.countplot(x='urgency', hue='urgency', data=df_result, ax=ax3,
                  palette="Reds", legend=False)
    ax3.set_title("Urgency Level")

    st.pyplot(fig)

    csv = df_result.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    st.markdown(f"### 📥 [Download Categorized Feedback CSV](data:file/csv;base64,{b64})", unsafe_allow_html=True)
else:
    st.warning("📤 Please upload a CSV file to begin analysis.")
