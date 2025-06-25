import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import altair as alt
from transformers import pipeline
from collections import Counter
import datetime
import base64

# Page setup
st.set_page_config(page_title="AI Feedback Categorizer", layout="wide")
st.title("🧠 AI-Powered Feedback Categorizer & Insight Generator")

# Load models
@st.cache_resource
def load_models():
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    sentiment_analyzer = pipeline("sentiment-analysis")
    return classifier, sentiment_analyzer

_classifier, _sentiment_analyzer = load_models()

# Helper functions
def assign_urgency(text):
    text = text.lower()
    if any(w in text for w in ["crash", "urgent", "fail", "error", "issue"]):
        return "High"
    elif any(w in text for w in ["slow", "delay", "not working"]):
        return "Medium"
    else:
        return "Low"

@st.cache_data
def process_feedback(df, _classifier, _sentiment_analyzer):
    labels = ["Bug", "Feature Request", "Compliment", "Complaint", "General Feedback"]

    df['category'] = df['feedback'].apply(lambda x: _classifier(x, labels)['labels'][0])
    df['confidence'] = df['feedback'].apply(lambda x: _classifier(x, labels)['scores'][0])
    df['sentiment'], df['sentiment_score'] = zip(*df['feedback'].apply(
        lambda x: (_sentiment_analyzer(x)[0]['label'], _sentiment_analyzer(x)[0]['score'])
    ))
    df['urgency'] = df['feedback'].apply(assign_urgency)
    df['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return df

@st.cache_data
def extract_keywords(text_series):
    all_words = " ".join(text_series).lower().split()
    stopwords = set(pd.read_csv("https://raw.githubusercontent.com/stopwords-iso/stopwords-en/master/stopwords-en.txt", header=None)[0])
    keywords = [word for word in all_words if word.isalpha() and word not in stopwords and len(word) > 3]
    return Counter(keywords).most_common(20)

# Upload section
uploaded_file = st.sidebar.file_uploader("📁 Upload a CSV with a 'feedback' column", type="csv")

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    if 'review' in df_raw.columns:
        df_raw.rename(columns={'review': 'feedback'}, inplace=True)

    if 'feedback' not in df_raw.columns:
        st.error("Uploaded file must have a 'feedback' column.")
        st.stop()

    with st.spinner("🔄 Analyzing feedback..."):
        df = process_feedback(df_raw.copy(), _classifier, _sentiment_analyzer)
        keyword_freq = extract_keywords(df['feedback'])

    # Tabs for navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard", "🧾 Data View", "📈 Keyword Trends", "💡 Business Insights"
    ])

    with tab1:
        st.subheader("Category Distribution")
        fig1, ax1 = plt.subplots()
        sns.countplot(y='category', data=df, ax=ax1, order=df['category'].value_counts().index, palette="viridis")
        st.pyplot(fig1)

        st.subheader("Sentiment Distribution")
        fig2, ax2 = plt.subplots()
        sns.countplot(x='sentiment', data=df, palette="Set2", ax=ax2)
        st.pyplot(fig2)

        st.subheader("Urgency Levels")
        fig3, ax3 = plt.subplots()
        sns.countplot(x='urgency', data=df, palette="Reds", ax=ax3)
        st.pyplot(fig3)

    with tab2:
        st.subheader("Processed Feedback Table")
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Processed Data", csv, "processed_feedback.csv", "text/csv")

    with tab3:
        st.subheader("Word Cloud")
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(keyword_freq))
        fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis("off")
        st.pyplot(fig_wc)

        st.subheader("Keyword Frequency Chart")
        keyword_df = pd.DataFrame(keyword_freq, columns=['keyword', 'count'])
        chart = alt.Chart(keyword_df).mark_bar().encode(
            x=alt.X('keyword', sort='-y'),
            y='count',
            tooltip=['keyword', 'count']
        ).properties(height=400)
        st.altair_chart(chart, use_container_width=True)

    with tab4:
        st.subheader("📊 Business Insights")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Feedback", len(df))
            st.metric("% Urgent", f"{(df['urgency']=='High').sum() / len(df) * 100:.2f}%")
        with col2:
            st.metric("Positive Sentiment", f"{(df['sentiment']=='POSITIVE').sum()} / {len(df)}")
            st.metric("Bugs Reported", (df['category']=='Bug').sum())

        st.write("### Weekly Summary")
        summary = df.groupby('category').agg({
            'sentiment': lambda x: x.value_counts().idxmax(),
            'urgency': lambda x: x.value_counts().idxmax(),
            'feedback': 'count'
        }).rename(columns={'feedback': 'count'}).reset_index()
        st.dataframe(summary)

else:
    st.info("Please upload a CSV file with a 'feedback' column to begin.")

st.caption("🚀 Built with Streamlit, Hugging Face Transformers, and ❤️")
