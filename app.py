import streamlit as st
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from io import StringIO
import time

# Only download required NLTK data with error handling
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    with st.spinner("Downloading NLTK resources..."):
        nltk.download('punkt')
        nltk.download('vader_lexicon')

# Set page config
st.set_page_config(
    page_title="Sentiment Analysis Tool",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling - optimized
def inject_custom_css():
    st.markdown("""
    <style>
    /* Main app styling */
    [data-testid="stAppViewContainer"] {
        background-color: #f8f9fa;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #6e8efb, #a777e3);
        color: white;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 8px 8px 0 0;
        margin-right: 5px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #6e8efb !important;
        color: white !important;
    }
    .stButton button {
        background-color: #6e8efb !important;
        border-radius: 8px !important;
        color: white !important;
        border: none !important;
    }
    .stButton button:hover {
        background-color: #5a7de3 !important;
    }
    .stTextArea textarea {
        border-radius: 8px !important;
        padding: 10px !important;
    }
    /* Metrics cards styling */
    [data-testid="metric-container"] {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    /* Progress bar styling */
    [data-testid="stProgress"] > div > div > div > div {
        background-color: #6e8efb;
    }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# Initialize sentiment analyzers with caching
@st.cache_resource
def load_analyzers():
    return SentimentIntensityAnalyzer()

sia = load_analyzers()

# Sample dataset - using smaller synthetic data
@st.cache_data
def load_sample_data():
    positive_samples = [
        "This product is amazing! I love it so much.",
        "Excellent quality and fast shipping. Highly recommend!",
        "Works perfectly, exactly as described.",
        "Very satisfied with my purchase. Will buy again.",
        "Great customer service and fantastic product.",
        "The best purchase I've made this year!",
        "Absolutely worth every penny. 5 stars!",
        "Exceeded all my expectations. Wonderful!",
        "Perfect in every way. Couldn't be happier.",
        "Quick delivery and superb quality."
    ]
    
    negative_samples = [
        "Terrible experience. Would not recommend.",
        "Poor quality and didn't work as expected.",
        "Waste of money. Product broke immediately.",
        "Very disappointed with this purchase.",
        "Awful customer service and bad product.",
        "Complete garbage. Don't waste your time.",
        "Not as advertised. False claims.",
        "Defective item received. Very unhappy.",
        "Horrible experience from start to finish.",
        "Regret buying this. Total scam."
    ]
    
    neutral_samples = [
        "It's okay, nothing special.",
        "The product works as expected.",
        "Average quality for the price.",
        "Neither good nor bad, just acceptable.",
        "Meets basic requirements, nothing more.",
        "Standard product, nothing exceptional.",
        "Does the job, but not impressed.",
        "Fair quality for the money spent.",
        "Basic functionality, as described.",
        "Not bad, but not great either."
    ]
    
    data = [{"text": text, "sentiment": "positive"} for text in positive_samples]
    data.extend([{"text": text, "sentiment": "negative"} for text in negative_samples])
    data.extend([{"text": text, "sentiment": "neutral"} for text in neutral_samples])
    
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
            "Worst experience ever.",
            "The service was acceptable.",
            "Beyond my wildest expectations!",
            "Complete waste of time and money.",
            "Average performance overall.",
            "Highly recommended with no reservations."
        ]
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This tool analyzes text sentiment using two approaches:
    1. **TextBlob**: Provides polarity (-1 to 1) and subjectivity (0 to 1) scores
    2. **VADER**: Specialized for social media/text with compound score (-1 to 1)
    """)

# Main content
st.title("📊 Sentiment Analysis Tool")
st.markdown("Analyze text sentiment quickly and efficiently using NLP techniques")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Single Text", "Batch Processing", "Sample Analysis"])

with tab1:
    st.header("Analyze Single Text")
    
    text_input = st.text_area("Input Text", sample_text, height=150)
    
    if st.button("Analyze Sentiment", key="analyze_single"):
        with st.spinner("Analyzing sentiment..."):
            time.sleep(0.5)  # Simulate processing time
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### TextBlob Analysis")
                blob = TextBlob(text_input)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                st.metric("Polarity", f"{polarity:.2f}", 
                          "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral",
                          delta_color="off")
                st.metric("Subjectivity", f"{subjectivity:.2f}", 
                          "Subjective" if subjectivity > 0.5 else "Objective",
                          delta_color="off")
                
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
                
                # VADER sentiment gauge
                fig, ax = plt.subplots(figsize=(8, 2))
                ax.set_xlim(-1, 1)
                ax.axvline(x=scores['compound'], color='#6e8efb', linewidth=5)
                ax.axvline(x=0, color='gray', linestyle='--')
                ax.set_yticks([])
                ax.set_title('Sentiment Intensity')
                ax.set_xlabel('Negative --------------------- Neutral --------------------- Positive')
                st.pyplot(fig)

with tab2:
    st.header("Batch Processing")
    
    # Option 1: Paste text
    st.markdown("### Option 1: Paste Text")
    batch_text = st.text_area("Enter multiple texts (one per line)", 
                            "I love this!\nThis is terrible.\nIt's okay.\nAbsolutely fantastic!\nWorst experience ever.",
                            height=150)
    
    if st.button("Analyze Pasted Text", key="analyze_pasted"):
        texts = [t.strip() for t in batch_text.split('\n') if t.strip()]
        
        if not texts:
            st.warning("Please enter some text to analyze")
        else:
            with st.spinner(f"Analyzing {len(texts)} texts..."):
                progress_bar = st.progress(0)
                results = []
                
                for i, text in enumerate(texts):
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
                    progress_bar.progress((i + 1) / len(texts))
                
                results_df = pd.DataFrame(results)
                st.success(f"Analysis complete for {len(texts)} texts!")
                
                # Display results
                st.dataframe(results_df.style.background_gradient(
                    subset=['textblob_polarity', 'vader_compound'],
                    cmap='RdYlGn',
                    vmin=-1,
                    vmax=1
                ))
                
                # Visualizations
                st.markdown("### Sentiment Distribution")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**TextBlob Polarity**")
                    fig, ax = plt.subplots()
                    sns.histplot(results_df['textblob_polarity'], bins=10, kde=True, ax=ax)
                    ax.set_xlim(-1, 1)
                    st.pyplot(fig)
                
                with col2:
                    st.markdown("**VADER Compound**")
                    fig, ax = plt.subplots()
                    sns.histplot(results_df['vader_compound'], bins=10, kde=True, ax=ax)
                    ax.set_xlim(-1, 1)
                    st.pyplot(fig)
                
                # Download
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "Download Results as CSV",
                    csv,
                    "sentiment_results.csv",
                    "text/csv",
                    key='download_csv'
                )
    
    # Option 2: Upload file
    st.markdown("### Option 2: Upload CSV")
    uploaded_file = st.file_uploader("Choose a CSV file (must contain a 'text' column)", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'text' not in df.columns:
                st.error("CSV must contain a 'text' column for analysis")
            else:
                st.write("Preview of uploaded data:")
                st.dataframe(df.head())
                
                if st.button("Analyze Uploaded File", key="analyze_uploaded"):
                    with st.spinner(f"Analyzing {len(df)} texts..."):
                        progress_bar = st.progress(0)
                        results = []
                        
                        for i, text in enumerate(df['text']):
                            text = str(text)  # Ensure text is string
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
                            progress_bar.progress((i + 1) / len(df['text'])))
                        
                        results_df = pd.DataFrame(results)
                        st.success("Analysis complete!")
                        st.dataframe(results_df.style.background_gradient(
                            subset=['textblob_polarity', 'vader_compound'],
                            cmap='RdYlGn',
                            vmin=-1,
                            vmax=1
                        ))
                        
                        # Visualizations
                        st.markdown("### Advanced Analysis")
                        
                        tab1, tab2, tab3 = st.tabs(["Distributions", "Word Clouds", "Correlations"])
                        
                        with tab1:
                            st.markdown("#### Sentiment Distributions")
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                            
                            # TextBlob polarity distribution
                            sns.histplot(results_df['textblob_polarity'], bins=15, kde=True, ax=ax1)
                            ax1.set_title('TextBlob Polarity Distribution')
                            ax1.set_xlim(-1, 1)
                            
                            # VADER compound distribution
                            sns.histplot(results_df['vader_compound'], bins=15, kde=True, ax=ax2)
                            ax2.set_title('VADER Compound Distribution')
                            ax2.set_xlim(-1, 1)
                            
                            st.pyplot(fig)
                        
                        with tab2:
                            st.markdown("#### Word Clouds by Sentiment")
                            
                            # Group by VADER sentiment
                            positive_texts = " ".join(results_df[results_df['vader_sentiment'] == 'Positive']['text'])
                            negative_texts = " ".join(results_df[results_df['vader_sentiment'] == 'Negative']['text'])
                            neutral_texts = " ".join(results_df[results_df['vader_sentiment'] == 'Neutral']['text'])
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown("**Positive**")
                                if positive_texts:
                                    wordcloud = WordCloud(width=300, height=200, background_color='white').generate(positive_texts)
                                    fig, ax = plt.subplots()
                                    ax.imshow(wordcloud, interpolation='bilinear')
                                    ax.axis('off')
                                    st.pyplot(fig)
                                else:
                                    st.info("No positive texts found")
                            
                            with col2:
                                st.markdown("**Negative**")
                                if negative_texts:
                                    wordcloud = WordCloud(width=300, height=200, background_color='white').generate(negative_texts)
                                    fig, ax = plt.subplots()
                                    ax.imshow(wordcloud, interpolation='bilinear')
                                    ax.axis('off')
                                    st.pyplot(fig)
                                else:
                                    st.info("No negative texts found")
                            
                            with col3:
                                st.markdown("**Neutral**")
                                if neutral_texts:
                                    wordcloud = WordCloud(width=300, height=200, background_color='white').generate(neutral_texts)
                                    fig, ax = plt.subplots()
                                    ax.imshow(wordcloud, interpolation='bilinear')
                                    ax.axis('off')
                                    st.pyplot(fig)
                                else:
                                    st.info("No neutral texts found")
                        
                        with tab3:
                            st.markdown("#### Correlation Between Metrics")
                            fig, ax = plt.subplots(figsize=(8, 6))
                            sns.scatterplot(
                                data=results_df,
                                x='textblob_polarity',
                                y='vader_compound',
                                hue='vader_sentiment',
                                palette={'Positive': 'green', 'Negative': 'red', 'Neutral': 'blue'},
                                ax=ax
                            )
                            ax.set_title('TextBlob vs VADER Sentiment Scores')
                            ax.set_xlabel('TextBlob Polarity')
                            ax.set_ylabel('VADER Compound')
                            ax.axhline(0, color='gray', linestyle='--')
                            ax.axvline(0, color='gray', linestyle='--')
                            st.pyplot(fig)
                        
                        # Download
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            "Download Full Results",
                            csv,
                            "sentiment_analysis_results.csv",
                            "text/csv",
                            key='download_full'
                        )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

with tab3:
    st.header("Sample Analysis")
    df = load_sample_data()
    
    st.markdown("### Sample Dataset (50 texts with known sentiment)")
    st.dataframe(df)
    
    st.markdown("### Visual Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Distributions", "Word Clouds", "Performance"])
    
    with tab1:
        st.markdown("#### Sentiment Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        df['sentiment'].value_counts().plot(kind='bar', color=['green', 'red', 'blue'], ax=ax)
        ax.set_title('Sentiment Distribution in Sample Data')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Count')
        st.pyplot(fig)
    
    with tab2:
        st.markdown("#### Word Cloud by Sentiment")
        
        sentiment_choice = st.selectbox("Select sentiment to visualize", ['positive', 'negative', 'neutral'])
        
        texts = " ".join(df[df['sentiment'] == sentiment_choice]['text'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(texts)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(f"Most Common Words in {sentiment_choice} Texts", pad=20)
        ax.axis('off')
        st.pyplot(fig)
    
    with tab3:
        st.markdown("#### Model Performance on Sample Data")
        
        if st.button("Evaluate Models", key="evaluate_models"):
            with st.spinner("Evaluating models..."):
                results = []
                
                for _, row in df.iterrows():
                    text = row['text']
                    true_sentiment = row['sentiment']
                    
                    # TextBlob analysis
                    blob = TextBlob(text)
                    tb_sentiment = "positive" if blob.sentiment.polarity > 0.1 else \
                                 "negative" if blob.sentiment.polarity < -0.1 else \
                                 "neutral"
                    
                    # VADER analysis
                    vader = sia.polarity_scores(text)
                    vader_sentiment = "positive" if vader['compound'] >= 0.05 else \
                                  "negative" if vader['compound'] <= -0.05 else \
                                  "neutral"
                    
                    results.append({
                        'text': text,
                        'true_sentiment': true_sentiment,
                        'textblob_prediction': tb_sentiment,
                        'vader_prediction': vader_sentiment
                    })
                
                results_df = pd.DataFrame(results)
                
                # Calculate accuracy
                tb_accuracy = (results_df['true_sentiment'] == results_df['textblob_prediction']).mean()
                vader_accuracy = (results_df['true_sentiment'] == results_df['vader_prediction']).mean()
                
                st.metric("TextBlob Accuracy", f"{tb_accuracy:.1%}")
                st.metric("VADER Accuracy", f"{vader_accuracy:.1%}")
                
                # Confusion matrices
                st.markdown("##### TextBlob Confusion Matrix")
                fig, ax = plt.subplots()
                pd.crosstab(
                    results_df['true_sentiment'], 
                    results_df['textblob_prediction'],
                    rownames=['Actual'],
                    colnames=['Predicted']
                ).plot(kind='bar', ax=ax)
                st.pyplot(fig)
                
                st.markdown("##### VADER Confusion Matrix")
                fig, ax = plt.subplots()
                pd.crosstab(
                    results_df['true_sentiment'], 
                    results_df['vader_prediction'],
                    rownames=['Actual'],
                    colnames=['Predicted']
                ).plot(kind='bar', ax=ax)
                st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
**Sentiment Analysis Tool**  
*Powered by TextBlob and NLTK's VADER*  
[Report issues](https://github.com/your-repo/issues) | [Contribute](https://github.com/your-repo)
""")
