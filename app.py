import streamlit as st
from textblob import TextBlob
import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import time

# Set page config - must be first Streamlit command
st.set_page_config(
    page_title="Sentiment Analysis Tool",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize sentiment analyzer with caching
@st.cache_resource
def load_analyzer():
    return SentimentIntensityAnalyzer()

sia = load_analyzer()

# Sample dataset
@st.cache_data
def load_sample_data():
    positive_samples = [
        "This product is amazing! I love it so much.",
        "Excellent quality and fast shipping. Highly recommend!",
        "Works perfectly, exactly as described.",
        "Very satisfied with my purchase. Will buy again.",
        "Great customer service and fantastic product."
    ]
    
    negative_samples = [
        "Terrible experience. Would not recommend.",
        "Poor quality and didn't work as expected.",
        "Waste of money. Product broke immediately.",
        "Very disappointed with this purchase.",
        "Awful customer service and bad product."
    ]
    
    data = [{"text": text, "sentiment": "positive"} for text in positive_samples]
    data.extend([{"text": text, "sentiment": "negative"} for text in negative_samples])
    
    return pd.DataFrame(data)

# Custom CSS
def inject_css():
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #6e8efb, #a777e3);
        color: white;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #6e8efb !important;
        color: white !important;
    }
    .stButton button {
        background-color: #6e8efb !important;
        color: white !important;
        border-radius: 8px !important;
    }
    [data-testid="stMetric"] {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

inject_css()

# Sidebar
with st.sidebar:
    st.title("Sentiment Analysis")
    st.markdown("""
    Analyze text sentiment using:
    - TextBlob (rule-based)
    - VADER (lexicon-based)
    """)
    
    sample_text = st.selectbox(
        "Try sample text:",
        [
            "I love this product!",
            "This is terrible.",
            "It's okay, nothing special.",
            "Absolutely fantastic!",
            "Worst experience ever."
        ]
    )

# Main content
st.title("📊 Sentiment Analysis Tool")
tab1, tab2, tab3 = st.tabs(["Single Text", "Batch Process", "Sample Data"])

with tab1:
    st.header("Analyze Single Text")
    text_input = st.text_area("Enter text", sample_text, height=150)
    
    if st.button("Analyze"):
        with st.spinner("Analyzing..."):
            time.sleep(0.5)  # Simulate processing
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("TextBlob Analysis")
                blob = TextBlob(text_input)
                st.metric("Polarity", f"{blob.sentiment.polarity:.2f}")
                st.metric("Subjectivity", f"{blob.sentiment.subjectivity:.2f}")
                
                if blob.sentiment.polarity > 0.1:
                    st.success("😊 Positive")
                elif blob.sentiment.polarity < -0.1:
                    st.error("😠 Negative")
                else:
                    st.info("😐 Neutral")
            
            with col2:
                st.subheader("VADER Analysis")
                scores = sia.polarity_scores(text_input)
                st.metric("Positive", f"{scores['pos']:.2f}")
                st.metric("Negative", f"{scores['neg']:.2f}")
                st.metric("Neutral", f"{scores['neu']:.2f}")
                st.metric("Compound", f"{scores['compound']:.2f}")
                
                if scores['compound'] >= 0.05:
                    st.success("😊 Positive")
                elif scores['compound'] <= -0.05:
                    st.error("☹️ Negative")
                else:
                    st.info("😐 Neutral")

with tab2:
    st.header("Batch Processing")
    batch_text = st.text_area("Enter multiple texts (one per line)", 
                            "I love this!\nThis is terrible.\nIt's okay.", 
                            height=150)
    
    if st.button("Analyze Batch"):
        texts = [t.strip() for t in batch_text.split('\n') if t.strip()]
        if texts:
            results = []
            progress_bar = st.progress(0)
            
            for i, text in enumerate(texts):
                blob = TextBlob(text)
                vader = sia.polarity_scores(text)
                
                results.append({
                    'text': text,
                    'tb_polarity': blob.sentiment.polarity,
                    'vader_compound': vader['compound'],
                    'vader_sentiment': "Positive" if vader['compound'] >= 0.05 else 
                                     "Negative" if vader['compound'] <= -0.05 else 
                                     "Neutral"
                })
                progress_bar.progress((i + 1) / len(texts))
            
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)
            
            # Visualization
            st.subheader("Sentiment Distribution")
            fig, ax = plt.subplots()
            results_df['vader_sentiment'].value_counts().plot(kind='bar', ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Please enter some text")

with tab3:
    st.header("Sample Data Analysis")
    df = load_sample_data()
    st.dataframe(df)
    
    st.subheader("Word Cloud")
    all_text = " ".join(df['text'])
    wordcloud = WordCloud(width=800, height=400).generate(all_text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

st.markdown("---")
st.caption("Sentiment Analysis Tool v1.0 | Powered by TextBlob and NLTK")
