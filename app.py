import streamlit as st
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import movie_reviews
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from wordcloud import WordCloud

# Download required NLTK data
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('movie_reviews')

# Set page config
st.set_page_config(
    page_title="Sentiment Analysis Tool",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
def inject_custom_css():
    st.markdown("""
    <style>
    /* Main content */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #6e8efb, #a777e3);
        color: white;
    }
    
    /* Sidebar header */
    [data-testid="stSidebar"] .sidebar-header {
        color: white;
        font-size: 24px;
        font-weight: bold;
        padding: 1rem;
        text-align: center;
    }
    
    /* Sidebar links */
    [data-testid="stSidebar"] .sidebar-content .stButton button {
        background-color: transparent;
        color: white;
        border: 1px solid white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
        width: 100%;
        text-align: left;
    }
    
    [data-testid="stSidebar"] .sidebar-content .stButton button:hover {
        background-color: rgba(255, 255, 255, 0.1);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 20px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0 !important;
        margin-right: 5px !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #6e8efb !important;
        color: white !important;
    }
    
    /* Cards */
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    /* Metrics */
    [data-testid="metric-container"] {
        border: 1px solid #e1e4e8;
        border-radius: 10px;
        padding: 15px;
        background-color: white;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #6e8efb !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
    }
    
    .stButton button:hover {
        background-color: #5a7de3 !important;
    }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# Initialize sentiment analyzers
sia = SentimentIntensityAnalyzer()

# Sample dataset
def load_sample_data():
    positive_reviews = [" ".join(movie_reviews.words(fileid)) for fileid in movie_reviews.fileids('pos')[:100]]
    negative_reviews = [" ".join(movie_reviews.words(fileid)) for fileid in movie_reviews.fileids('neg')[:100]]
    
    data = []
    for review in positive_reviews:
        data.append({
            "text": review[:500] + "...",  # Truncate long reviews
            "sentiment": "positive"
        })
    
    for review in negative_reviews:
        data.append({
            "text": review[:500] + "...",
            "sentiment": "negative"
        })
    
    return pd.DataFrame(data)

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-header">Sentiment Analysis Tool</div>', unsafe_allow_html=True)
    st.markdown("""
    Analyze text sentiment using:
    - TextBlob (rule-based)
    - NLTK's VADER (lexicon-based)
    """)
    
    st.markdown("### Features")
    st.markdown("""
    - Real-time sentiment analysis
    - Batch processing
    - Visualization
    - Sample dataset exploration
    """)
    
    st.markdown("### About")
    st.markdown("""
    This app uses natural language processing to determine the emotional tone behind text.
    """)

# Main content
st.title("📊 Sentiment Analysis Tool")
st.markdown("Analyze text sentiment using different NLP techniques")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Single Text", "Batch Processing", "Visualization", "Sample Dataset"])

with tab1:
    st.header("Analyze Single Text")
    st.markdown("Enter text below to analyze its sentiment")
    
    text_input = st.text_area("Input Text", "I love this product! It's amazing and works perfectly.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### TextBlob Analysis")
        if st.button("Analyze with TextBlob"):
            blob = TextBlob(text_input)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            st.metric("Polarity", f"{polarity:.2f}", 
                      "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral")
            st.metric("Subjectivity", f"{subjectivity:.2f}", 
                      "Subjective" if subjectivity > 0.5 else "Objective")
            
            # Interpretation
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
        if st.button("Analyze with VADER"):
            scores = sia.polarity_scores(text_input)
            
            st.metric("Positive", f"{scores['pos']:.2f}")
            st.metric("Negative", f"{scores['neg']:.2f}")
            st.metric("Neutral", f"{scores['neu']:.2f}")
            st.metric("Compound", f"{scores['compound']:.2f}")
            
            # Interpretation
            if scores['compound'] >= 0.05:
                sentiment = "😊 Positive"
            elif scores['compound'] <= -0.05:
                sentiment = "☹️ Negative"
            else:
                sentiment = "😐 Neutral"
            
            st.markdown(f"**Overall Sentiment:** {sentiment}")

with tab2:
    st.header("Batch Processing")
    st.markdown("Upload a CSV file with text to analyze (needs a 'text' column)")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())
        
        if 'text' not in df.columns:
            st.error("The uploaded file must contain a 'text' column")
        else:
            if st.button("Analyze All Texts"):
                progress_bar = st.progress(0)
                results = []
                
                for i, row in df.iterrows():
                    text = row['text']
                    
                    # TextBlob analysis
                    blob = TextBlob(str(text))
                    polarity = blob.sentiment.polarity
                    subjectivity = blob.sentiment.subjectivity
                    
                    # VADER analysis
                    vader_scores = sia.polarity_scores(str(text))
                    
                    results.append({
                        'text': text,
                        'textblob_polarity': polarity,
                        'textblob_subjectivity': subjectivity,
                        'vader_pos': vader_scores['pos'],
                        'vader_neg': vader_scores['neg'],
                        'vader_neu': vader_scores['neu'],
                        'vader_compound': vader_scores['compound']
                    })
                    
                    progress_bar.progress((i + 1) / len(df))
                
                results_df = pd.DataFrame(results)
                st.success("Analysis complete!")
                st.dataframe(results_df)
                
                # Download button
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Results",
                    csv,
                    "sentiment_analysis_results.csv",
                    "text/csv",
                    key='download-csv'
                )

with tab3:
    st.header("Visualization")
    st.markdown("Visualize sentiment analysis results")
    
    # Sample visualization with random data
    st.markdown("### Sample Word Cloud")
    text = " ".join(movie_reviews.words()[:1000])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
    
    # Sentiment distribution
    st.markdown("### Sentiment Distribution")
    sample_data = load_sample_data()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=sample_data, x='sentiment', ax=ax, palette='viridis')
    ax.set_title("Distribution of Sentiments in Sample Data")
    st.pyplot(fig)
    
    # Comparison of analyzers
    st.markdown("### Analyzer Comparison")
    sample_texts = [
        "I love this product! It's amazing.",
        "This is the worst experience ever.",
        "The item is okay, nothing special.",
        "Absolutely fantastic service and quality!",
        "Terrible customer support and bad quality."
    ]
    
    comparison_data = []
    for text in sample_texts:
        blob = TextBlob(text)
        vader = sia.polarity_scores(text)
        
        comparison_data.append({
            'text': text,
            'textblob': blob.sentiment.polarity,
            'vader': vader['compound']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    comparison_df.set_index('text').plot(kind='bar', ax=ax, color=['#6e8efb', '#a777e3'])
    ax.set_title("TextBlob vs VADER Sentiment Scores")
    ax.set_ylabel("Sentiment Score")
    ax.set_xlabel("Text")
    st.pyplot(fig)

with tab4:
    st.header("Sample Dataset Exploration")
    st.markdown("Explore the NLTK movie reviews dataset")
    
    df = load_sample_data()
    st.dataframe(df.head())
    
    # Show random samples
    st.markdown("### Random Samples")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Positive Review**")
        positive_sample = random.choice(df[df['sentiment'] == 'positive']['text'].tolist())
        st.info(positive_sample)
    
    with col2:
        st.markdown("**Negative Review**")
        negative_sample = random.choice(df[df['sentiment'] == 'negative']['text'].tolist())
        st.error(negative_sample)
    
    # Statistics
    st.markdown("### Dataset Statistics")
    st.write(f"Total reviews: {len(df)}")
    st.write(f"Positive reviews: {len(df[df['sentiment'] == 'positive'])}")
    st.write(f"Negative reviews: {len(df[df['sentiment'] == 'negative'])}")
    
    # Word counts
    st.markdown("### Word Count Analysis")
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    sns.histplot(data=df, x='word_count', hue='sentiment', bins=20, ax=ax[0], palette='viridis')
    ax[0].set_title("Word Count Distribution by Sentiment")
    
    sns.boxplot(data=df, x='sentiment', y='word_count', ax=ax[1], palette='viridis')
    ax[1].set_title("Word Count by Sentiment")
    
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
Built with ❤️ using Streamlit, TextBlob, and NLTK  
[GitHub Repository](#) | [Report Issue](#)
""")
