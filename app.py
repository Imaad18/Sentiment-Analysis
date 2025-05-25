import streamlit as st
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from wordcloud import WordCloud
from io import StringIO

# Only download minimal required NLTK data
nltk.download('punkt')
nltk.download('vader_lexicon')

# Set page config
st.set_page_config(
    page_title="Sentiment Analysis Tool",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling - more optimized
def inject_custom_css():
    st.markdown("""
    <style>
    /* Simplified CSS for faster rendering */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #6e8efb, #a777e3);
        color: white;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0 15px;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #6e8efb !important;
    }
    .stButton button {
        background-color: #6e8efb !important;
        border-radius: 8px !important;
    }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# Initialize sentiment analyzers with caching
@st.cache_resource
def load_analyzers():
    return SentimentIntensityAnalyzer()

sia = load_analyzers()

# Sample dataset - using smaller synthetic data instead of movie_reviews
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

# Sidebar
with st.sidebar:
    st.markdown("## Sentiment Analysis Tool")
    st.markdown("""
    Analyze text sentiment using:
    - TextBlob (rule-based)
    - NLTK's VADER (lexicon-based)
    """)
    
    st.markdown("### Quick Analysis")
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
st.markdown("Analyze text sentiment quickly and efficiently")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Single Text", "Batch Processing", "Sample Analysis"])

with tab1:
    st.header("Analyze Single Text")
    
    text_input = st.text_area("Input Text", sample_text)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### TextBlob Analysis")
        blob = TextBlob(text_input)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        st.metric("Polarity", f"{polarity:.2f}", 
                  "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral")
        st.metric("Subjectivity", f"{subjectivity:.2f}", 
                  "Subjective" if subjectivity > 0.5 else "Objective")
        
        if polarity > 0.5:
            sentiment = "😊 Strongly Positive"
        elif polarity > 0.1:
            sentiment = "🙂 Positive"
        elif polarity < -0.5:
            sentiment = "😠 Strongly Negative"
        elif polarity < -0.1:
            sentiment = "☹️ Negative"
        else:
            sentiment = "😐 Neutral"
        
        st.markdown(f"**Overall Sentiment:** {sentiment}")
    
    with col2:
        st.markdown("### VADER Analysis")
        scores = sia.polarity_scores(text_input)
        
        st.metric("Positive", f"{scores['pos']:.2f}")
        st.metric("Negative", f"{scores['neg']:.2f}")
        st.metric("Neutral", f"{scores['neu']:.2f}")
        st.metric("Compound", f"{scores['compound']:.2f}")
        
        if scores['compound'] >= 0.05:
            sentiment = "😊 Positive"
        elif scores['compound'] <= -0.05:
            sentiment = "☹️ Negative"
        else:
            sentiment = "😐 Neutral"
        
        st.markdown(f"**Overall Sentiment:** {sentiment}")

with tab2:
    st.header("Batch Processing")
    
    # Option 1: Paste text
    st.markdown("### Option 1: Paste Text")
    batch_text = st.text_area("Enter multiple texts (one per line)", 
                            "I love this!\nThis is terrible.\nIt's okay.")
    
    if st.button("Analyze Pasted Text"):
        texts = [t.strip() for t in batch_text.split('\n') if t.strip()]
        results = []
        
        for text in texts:
            blob = TextBlob(text)
            vader = sia.polarity_scores(text)
            
            results.append({
                'text': text,
                'textblob_polarity': blob.sentiment.polarity,
                'textblob_subjectivity': blob.sentiment.subjectivity,
                'vader_compound': vader['compound'],
                'vader_sentiment': "Positive" if vader['compound'] >= 0.05 else 
                                 "Negative" if vader['compound'] <= -0.05 else 
                                 "Neutral"
            })
        
        results_df = pd.DataFrame(results)
        st.dataframe(results_df)
        
        # Simple visualization
        fig, ax = plt.subplots()
        results_df['vader_sentiment'].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)
    
    # Option 2: Upload file
    st.markdown("### Option 2: Upload CSV")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'text' not in df.columns:
                st.error("CSV must contain a 'text' column")
            else:
                st.write("Preview:")
                st.dataframe(df.head())
                
                if st.button("Analyze Uploaded File"):
                    with st.spinner("Analyzing..."):
                        results = []
                        for text in df['text']:
                            blob = TextBlob(str(text))
                            vader = sia.polarity_scores(str(text))
                            results.append({
                                'text': text,
                                'textblob_polarity': blob.sentiment.polarity,
                                'vader_compound': vader['compound'],
                                'vader_sentiment': "Positive" if vader['compound'] >= 0.05 else 
                                                 "Negative" if vader['compound'] <= -0.05 else 
                                                 "Neutral"
                            })
                        
                        results_df = pd.DataFrame(results)
                        st.success("Analysis complete!")
                        st.dataframe(results_df)
                        
                        # Download
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            "Download Results",
                            csv,
                            "sentiment_results.csv",
                            "text/csv"
                        )

        except Exception as e:
            st.error(f"Error reading file: {e}")

with tab3:
    st.header("Sample Analysis")
    df = load_sample_data()
    
    st.markdown("### Sample Dataset")
    st.dataframe(df)
    
    st.markdown("### Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Sentiment Distribution**")
        fig, ax = plt.subplots()
        df['sentiment'].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)
    
    with col2:
        st.markdown("**Word Cloud**")
        all_text = " ".join(df['text'])
        wordcloud = WordCloud(width=400, height=200).generate(all_text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    
    st.markdown("### Try Sample Analysis")
    sample = st.selectbox("Choose a sample text", df['text'])
    
    if sample:
        st.write(f"**Selected Text:** {sample}")
        
        blob = TextBlob(sample)
        vader = sia.polarity_scores(sample)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("TextBlob Polarity", f"{blob.sentiment.polarity:.2f}")
        with col2:
            st.metric("VADER Compound", f"{vader['compound']:.2f}")

# Footer
st.markdown("---")
st.markdown("""
*Optimized sentiment analysis tool - faster loading and deployment*
""")
