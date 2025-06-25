# 🧠 AI Feedback Categorizer & Insight Generator

An AI-powered Streamlit app that classifies user feedback, detects sentiment and urgency, extracts top keywords, and generates actionable business insights — built with Hugging Face Transformers and zero-shot classification.

## 📌 Problem Statement

In a world where companies receive massive amounts of unstructured user feedback, it's difficult to extract actionable insights manually. This project automates the analysis of customer feedback using modern LLM-based NLP models and presents it via an interactive dashboard.

---

## 🚀 Features

| Category | Feature |
|----------|---------|
| ✅ Core | Zero-shot classification of feedback into categories |
| ✅ Core | Sentiment analysis (Positive / Negative) |
| ✅ Core | Urgency scoring (High / Medium / Low) |
| ✅ Core | Keyword extraction from feedback |
| ✅ Core | WordCloud and frequency bar chart |
| ✅ Core | Data table with filters + CSV download |
| ✅ Core | Business Insights tab with KPIs and summaries |
| ✅ Core | Interactive charts using Matplotlib, Seaborn, Altair |
| ✅ Enhancements | Multilingual feedback handling |
| ✅ Enhancements | Real-time alert system for negative/urgent feedback |
| ✅ Enhancements | Cloud deployment ready (Streamlit Cloud / Hugging Face Spaces) |

---

## 🧠 Tech Stack

- **Python 3.10+**
- **Streamlit**
- **Hugging Face Transformers**
- **Facebook BART (Zero-shot)**
- **DistilBERT Sentiment Classifier**
- **Matplotlib / Seaborn / Altair / WordCloud**
- **Pandas / Numpy / Counter**
- **Deployed on Hugging Face Spaces / Streamlit Cloud**

---

## 📊 How It Works

1. **Upload CSV** file with a `feedback` column.
2. App classifies each feedback using:
   - Zero-shot classification (`facebook/bart-large-mnli`)
   - Sentiment analysis (`distilbert-base-uncased-finetuned-sst-2-english`)
   - Urgency detection based on keywords
3. Generates:
   - Feedback summary table
   - Charts for category, sentiment, urgency
   - Top keywords via WordCloud + Bar Chart
   - Business insights summary

---

## 📂 Folder Structure

