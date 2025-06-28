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
import io

st.set_page_config(page_title="AI Feedback Analyzer", layout="wide")
st.title("🧠 AI-Powered Feedback Categorizer & Business Insights")

# Model toggle
use_multilingual = st.sidebar.toggle("🌐 Use Multilingual Model (slower)", value=False)

@st.cache_resource
def load_models(use_multi=False):
    model_name = "joeddav/xlm-roberta-large-xnli" if use_multi else "facebook/bart-large-mnli"
    classifier = pipeline("zero-shot-classification", model=model_name)
    sentiment_analyzer = pipeline("sentiment-analysis")
    return classifier, sentiment_analyzer

classifier, sentiment_analyzer = load_models(use_multilingual)

def classify_feedback(feedback, labels):
    result = classifier(feedback, labels)
    return result['labels'][0], result['scores'][0]

def analyze_sentiment(text):
    result = sentiment_analyzer(text)[0]
    return result['label'], result['score']

def assign_urgency(feedback):
    feedback = feedback.lower()
    if any(w in feedback for w in ["crash", "urgent", "fail", "error", "issue"]):
        return "High"
    elif any(w in feedback for w in ["slow", "delay", "not working"]):
        return "Medium"
    return "Low"

def extract_keywords(texts):
    all_words = " ".join(texts).lower().split()
    stopwords = set(pd.read_csv("https://raw.githubusercontent.com/stopwords-iso/stopwords-en/master/stopwords-en.txt", header=None)[0].tolist())
    keywords = [w for w in all_words if w.isalpha() and w not in stopwords and len(w) > 3]
    return Counter(keywords).most_common(20)

uploaded_file = st.sidebar.file_uploader("📁 Upload CSV file with 'feedback' column", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'review' in df.columns:
        df.rename(columns={'review': 'feedback'}, inplace=True)
    if 'feedback' not in df.columns:
        st.error("CSV must contain a 'feedback' column.")
    else:
        with st.spinner("⚙️ Processing..."):
            labels = ["Bug", "Feature Request", "Compliment", "Complaint", "General Feedback"]
            df['category'] = df['feedback'].apply(lambda x: classify_feedback(x, labels)[0])
            df['confidence'] = df['feedback'].apply(lambda x: classify_feedback(x, labels)[1])
            df['sentiment'], df['sentiment_score'] = zip(*df['feedback'].apply(analyze_sentiment))
            df['urgency'] = df['feedback'].apply(assign_urgency)
            df['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            keyword_freq = extract_keywords(df['feedback'])
            df['month'] = datetime.datetime.now().strftime("%Y-%m")

        tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📁 Data", "📈 Trends", "💼 Business Insights"])

        with tab1:
            st.subheader("Category Distribution")
            fig1, ax1 = plt.subplots()
            sns.countplot(y='category', data=df, ax=ax1, order=df['category'].value_counts().index, palette="viridis")
            st.pyplot(fig1)

            st.subheader("Sentiment Distribution")
            fig2, ax2 = plt.subplots()
            sns.countplot(x='sentiment', data=df, ax=ax2, palette="Set2")
            st.pyplot(fig2)

            st.subheader("Urgency Levels")
            fig3, ax3 = plt.subplots()
            sns.countplot(x='urgency', data=df, ax=ax3, palette="Reds")
            st.pyplot(fig3)

        with tab2:
            st.subheader("📄 Processed Feedback Data")
            st.dataframe(df)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV", csv, "processed_feedback.csv", "text/csv")

        with tab3:
            st.subheader("Top Keywords")
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(keyword_freq))
            fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
            ax_wc.imshow(wordcloud, interpolation='bilinear')
            ax_wc.axis("off")
            st.pyplot(fig_wc)

            st.subheader("Keyword Frequency Chart")
            keyword_df = pd.DataFrame(keyword_freq, columns=["keyword", "count"])
            chart = alt.Chart(keyword_df).mark_bar().encode(
                x=alt.X("keyword", sort='-y'),
                y="count",
                tooltip=["keyword", "count"]
            ).properties(height=400)
            st.altair_chart(chart, use_container_width=True)

        with tab4:
            st.subheader("📊 Business Insights")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Feedback", len(df))
                st.metric("% Urgent", f"{round((df['urgency']=='High').sum()/len(df)*100,2)}%")
            with col2:
                st.metric("Positive Reviews", f"{(df['sentiment']=='POSITIVE').sum()} / {len(df)}")
                st.metric("Bugs Found", (df['category']=='Bug').sum())

            st.write("### Weekly Summary")
            summary = df.groupby('category').agg({
                'sentiment': lambda x: x.value_counts().idxmax(),
                'urgency': lambda x: x.value_counts().idxmax(),
                'feedback': 'count'
            }).rename(columns={'feedback': 'count'}).reset_index()
            st.dataframe(summary)

else:
    st.info("Please upload a CSV file with a 'feedback' or 'review' column to begin.")

st.caption("🛠 Built using Streamlit • Transformers • Hugging Face • Altair • WordCloud")
