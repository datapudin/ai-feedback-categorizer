import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import altair as alt
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from collections import Counter
import datetime
import base64
import io
import time

# Set Streamlit config
st.set_page_config(page_title="AI Feedback Categorizer", layout="wide")
st.title("🧠 AI Feedback Categorizer & Business Insights")

# Load models
@st.cache_resource
def load_models():
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return classifier, sentiment_analyzer

classifier, sentiment_analyzer = load_models()

# Classify feedback into categories
def classify_feedback(feedback, labels):
    result = classifier(feedback, labels)
    return result['labels'][0], result['scores'][0]

# Sentiment analysis
def analyze_sentiment(text):
    result = sentiment_analyzer(text)[0]
    return result['label'], result['score']

# Assign urgency
def assign_urgency(feedback):
    keywords = feedback.lower()
    if any(k in keywords for k in ["crash", "fail", "urgent", "bug", "not opening"]):
        return "High"
    elif any(k in keywords for k in ["slow", "lag", "delay"]):
        return "Medium"
    else:
        return "Low"

# Extract keywords
def extract_keywords(texts):
    all_words = " ".join(texts).lower().split()
    stopwords = set(pd.read_csv("https://raw.githubusercontent.com/stopwords-iso/stopwords-en/master/stopwords-en.txt", header=None)[0].tolist())
    keywords = [word for word in all_words if word.isalpha() and word not in stopwords and len(word) > 3]
    return Counter(keywords).most_common(20)

# Keyword trends over time
def get_keyword_trends(df):
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    keywords_daily = {}
    for date, group in df.groupby('date'):
        all_words = " ".join(group['feedback']).lower().split()
        filtered = [w for w in all_words if w.isalpha() and len(w) > 3]
        top = Counter(filtered).most_common(5)
        keywords_daily[date] = [w for w, _ in top]
    return keywords_daily

# Alert system
def show_alerts(df):
    urgent_count = (df['urgency'] == "High").sum()
    negative_count = (df['sentiment'] == "NEGATIVE").sum()
    if urgent_count > 5:
        st.toast(f"🚨 High number of urgent issues: {urgent_count}")
    if negative_count > 5:
        st.toast(f"😡 Surge in negative feedback: {negative_count}")

# File uploader
uploaded_file = st.sidebar.file_uploader("📁 Upload CSV with 'feedback' column", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'review' in df.columns:
        df.rename(columns={'review': 'feedback'}, inplace=True)

    if 'feedback' not in df.columns:
        st.error("❌ File must contain a 'feedback' column.")
    else:
        with st.spinner("🔍 Analyzing feedback..."):
            time.sleep(1)
            labels = ["Bug", "Feature Request", "Compliment", "Complaint", "General Feedback"]
            df['category'], df['confidence'] = zip(*df['feedback'].apply(lambda x: classify_feedback(x, labels)))
            df['sentiment'], df['sentiment_score'] = zip(*df['feedback'].apply(analyze_sentiment))
            df['urgency'] = df['feedback'].apply(assign_urgency)
            df['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            keyword_freq = extract_keywords(df['feedback'])
            show_alerts(df)

        tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📁 Data", "🔤 Keyword Trends", "📈 Business Insights"])

        with tab1:
            st.subheader("📌 Category Breakdown")
            fig1, ax1 = plt.subplots()
            sns.countplot(y='category', data=df, ax=ax1, order=df['category'].value_counts().index, palette="viridis")
            st.pyplot(fig1)

            st.subheader("🧭 Sentiment Overview")
            fig2, ax2 = plt.subplots()
            sns.countplot(x='sentiment', data=df, ax=ax2, palette="Set2")
            st.pyplot(fig2)

            st.subheader("🔥 Urgency Levels")
            fig3, ax3 = plt.subplots()
            sns.countplot(x='urgency', data=df, ax=ax3, palette="Reds")
            st.pyplot(fig3)

        with tab2:
            st.subheader("📁 Raw Feedback Data")
            st.dataframe(df)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download Results", csv, "processed_feedback.csv", "text/csv")

        with tab3:
            st.subheader("☁️ Word Cloud")
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(dict(keyword_freq))
            fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
            ax_wc.imshow(wordcloud, interpolation="bilinear")
            ax_wc.axis("off")
            st.pyplot(fig_wc)

            st.subheader("📅 Top Keywords Over Time")
            trends = get_keyword_trends(df)
            trend_df = pd.DataFrame([
                {"date": date, "keyword": kw}
                for date, kws in trends.items()
                for kw in kws
            ])
            chart = alt.Chart(trend_df).mark_bar().encode(
                x="date:T",
                y="count()",
                color="keyword",
                tooltip=["keyword", "count()"]
            ).properties(height=400)
            st.altair_chart(chart, use_container_width=True)

        with tab4:
            st.subheader("💡 Business Insights Dashboard")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Feedback", len(df))
                st.metric("Urgent Issues", f"{(df['urgency']=='High').sum()} / {len(df)}")
            with col2:
                st.metric("Positive Feedback", f"{(df['sentiment']=='POSITIVE').sum()} / {len(df)}")
                st.metric("Bugs Reported", (df['category']=='Bug').sum())

            st.markdown("### 📅 Weekly Summary")
            summary = df.groupby(['category']).agg({
                'feedback': 'count',
                'sentiment': lambda x: x.value_counts().idxmax(),
                'urgency': lambda x: x.value_counts().idxmax()
            }).rename(columns={'feedback': 'count'}).reset_index()
            st.dataframe(summary)

            st.markdown("### 📌 Strategic Notes")
            st.markdown("""
            - 🚀 **Prioritize bug fixes** in 'High urgency' with negative sentiment.
            - 💡 **Leverage compliments** in marketing or testimonials.
            - 🔄 **Track trends** of repeated keywords to plan features or improvements.
            """)

else:
    st.info("📂 Upload a CSV file to start analysis.")

st.caption("Built with ❤️ using Streamlit, Hugging Face Transformers, and Matplotlib")
