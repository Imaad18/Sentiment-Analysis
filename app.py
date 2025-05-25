import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time
import re
from collections import Counter

# Set page config - must be first Streamlit command
st.set_page_config(
    page_title="Sentiment Analysis Tool",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple sentiment analysis functions (no external dependencies)
def simple_sentiment_analysis(text):
    """Simple rule-based sentiment analysis"""
    text = text.lower()
    
    # Positive words
    positive_words = [
        'good', 'great', 'excellent', 'amazing', 'fantastic', 'wonderful', 'awesome',
        'love', 'like', 'happy', 'satisfied', 'perfect', 'best', 'brilliant',
        'outstanding', 'superb', 'magnificent', 'terrific', 'fabulous', 'recommend'
    ]
    
    # Negative words
    negative_words = [
        'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'worst', 'poor',
        'disappointing', 'disappointed', 'useless', 'waste', 'broken', 'defective',
        'annoying', 'frustrating', 'ridiculous', 'pathetic', 'disgusting'
    ]
    
    # Count positive and negative words
    words = re.findall(r'\b\w+\b', text)
    pos_count = sum(1 for word in words if word in positive_words)
    neg_count = sum(1 for word in words if word in negative_words)
    
    # Calculate sentiment score
    total_sentiment_words = pos_count + neg_count
    if total_sentiment_words == 0:
        return 0.0, 0.5  # neutral sentiment, medium subjectivity
    
    polarity = (pos_count - neg_count) / len(words) * 10  # Scale to similar range as TextBlob
    subjectivity = total_sentiment_words / len(words) if words else 0
    
    return polarity, min(subjectivity, 1.0)

def advanced_sentiment_analysis(text):
    """More advanced rule-based sentiment with intensifiers and negations"""
    text = text.lower()
    
    # Sentiment lexicon with scores
    sentiment_words = {
        # Positive words
        'excellent': 3, 'amazing': 3, 'fantastic': 3, 'wonderful': 3, 'outstanding': 3,
        'great': 2, 'good': 2, 'nice': 2, 'happy': 2, 'satisfied': 2, 'perfect': 2,
        'love': 2, 'awesome': 2, 'brilliant': 2, 'superb': 2, 'terrific': 2,
        'like': 1, 'okay': 1, 'fine': 1, 'decent': 1, 'alright': 1,
        
        # Negative words
        'terrible': -3, 'awful': -3, 'horrible': -3, 'disgusting': -3, 'pathetic': -3,
        'bad': -2, 'poor': -2, 'worst': -2, 'hate': -2, 'disappointed': -2,
        'disappointing': -2, 'useless': -2, 'broken': -2, 'defective': -2,
        'dislike': -1, 'annoying': -1, 'frustrating': -1, 'boring': -1
    }
    
    # Intensifiers
    intensifiers = {
        'very': 1.5, 'extremely': 2.0, 'incredibly': 2.0, 'absolutely': 1.8,
        'really': 1.3, 'quite': 1.2, 'pretty': 1.1, 'so': 1.4, 'totally': 1.6
    }
    
    # Negations
    negations = ['not', 'no', 'never', 'nothing', 'nowhere', 'neither', 'nobody', 'none']
    
    words = re.findall(r'\b\w+\b', text)
    total_score = 0
    sentiment_word_count = 0
    
    for i, word in enumerate(words):
        if word in sentiment_words:
            score = sentiment_words[word]
            sentiment_word_count += 1
            
            # Check for intensifiers before the sentiment word
            if i > 0 and words[i-1] in intensifiers:
                score *= intensifiers[words[i-1]]
            
            # Check for negations (within 2 words before)
            negated = False
            for j in range(max(0, i-2), i):
                if words[j] in negations:
                    negated = True
                    break
            
            if negated:
                score *= -0.5
            
            total_score += score
    
    # Normalize scores
    if sentiment_word_count == 0:
        return {'compound': 0.0, 'pos': 0.33, 'neu': 0.34, 'neg': 0.33}
    
    # Calculate compound score (normalized between -1 and 1)
    compound = total_score / len(words) if words else 0
    compound = max(-1, min(1, compound))  # Clamp between -1 and 1
    
    # Calculate pos, neu, neg scores
    if compound >= 0.05:
        pos = 0.5 + (compound * 0.5)
        neg = 0.1
        neu = 1 - pos - neg
    elif compound <= -0.05:
        neg = 0.5 + (abs(compound) * 0.5)
        pos = 0.1
        neu = 1 - pos - neg
    else:
        neu = 0.7
        pos = 0.15
        neg = 0.15
    
    return {'compound': compound, 'pos': pos, 'neu': neu, 'neg': neg}

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
        border: 1px solid #e9ecef;
    }
    .metric-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

def create_simple_wordcloud(texts):
    """Create a simple word frequency visualization"""
    # Combine all texts
    all_text = " ".join(texts).lower()
    
    # Remove common words and punctuation
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                  'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
                  'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                  'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 
                  'they', 'my', 'your', 'his', 'her', 'its', 'our', 'their'}
    
    words = re.findall(r'\b\w+\b', all_text)
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Count word frequencies
    word_freq = Counter(filtered_words)
    
    return word_freq.most_common(20)

inject_css()

# Sidebar
with st.sidebar:
    st.title("🎭 Sentiment Analysis")
    st.markdown("""
    Analyze text sentiment using:
    - Simple Rule-Based Analysis
    - Advanced Lexicon Analysis
    
    *No external dependencies required!*
    """)
    
    sample_text = st.selectbox(
        "Try sample text:",
        [
            "I love this product! It's absolutely amazing!",
            "This is terrible and disappointing.",
            "It's okay, nothing special really.",
            "Absolutely fantastic! Highly recommend!",
            "Worst experience ever. Very frustrated."
        ]
    )

# Main content
st.title("📊 Sentiment Analysis Tool")
st.markdown("*Analyze text sentiment without external dependencies*")

tab1, tab2, tab3 = st.tabs(["Single Text Analysis", "Batch Processing", "Sample Data Analysis"])

with tab1:
    st.header("🔍 Analyze Single Text")
    text_input = st.text_area("Enter text to analyze:", value=sample_text, height=150)
    
    if st.button("🚀 Analyze Sentiment", type="primary"):
        if text_input.strip():
            with st.spinner("Analyzing sentiment..."):
                time.sleep(0.5)  # Simulate processing
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📝 Simple Analysis")
                    polarity, subjectivity = simple_sentiment_analysis(text_input)
                    
                    with st.container():
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                        st.metric("Polarity", f"{polarity:.3f}", help="Range: -1 (negative) to 1 (positive)")
                        st.metric("Subjectivity", f"{subjectivity:.3f}", help="Range: 0 (objective) to 1 (subjective)")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    if polarity > 0.1:
                        st.success("😊 **Positive Sentiment**")
                    elif polarity < -0.1:
                        st.error("😠 **Negative Sentiment**")
                    else:
                        st.info("😐 **Neutral Sentiment**")
                
                with col2:
                    st.subheader("🎯 Advanced Analysis")
                    scores = advanced_sentiment_analysis(text_input)
                    
                    with st.container():
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                        st.metric("Compound", f"{scores['compound']:.3f}", help="Overall sentiment score")
                        st.metric("Positive", f"{scores['pos']:.3f}")
                        st.metric("Neutral", f"{scores['neu']:.3f}")
                        st.metric("Negative", f"{scores['neg']:.3f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    if scores['compound'] >= 0.05:
                        st.success("😊 **Positive Sentiment**")
                    elif scores['compound'] <= -0.05:
                        st.error("😞 **Negative Sentiment**")
                    else:
                        st.info("😐 **Neutral Sentiment**")
                
                # Sentiment visualization
                st.subheader("📊 Sentiment Breakdown")
                chart_data = pd.DataFrame({
                    'Sentiment': ['Positive', 'Neutral', 'Negative'],
                    'Score': [scores['pos'], scores['neu'], scores['neg']]
                })
                st.bar_chart(chart_data.set_index('Sentiment'))
        else:
            st.warning("⚠️ Please enter some text to analyze")

with tab2:
    st.header("📋 Batch Processing")
    st.markdown("Analyze multiple texts at once (one per line)")
    
    batch_text = st.text_area(
        "Enter multiple texts (one per line):", 
        value="I love this product!\nThis is terrible quality.\nIt's okay, decent value.\nAbsolutely fantastic experience!\nVery disappointed with service.", 
        height=150
    )
    
    if st.button("🔄 Analyze All Texts", type="primary"):
        texts = [t.strip() for t in batch_text.split('\n') if t.strip()]
        if texts:
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, text in enumerate(texts):
                status_text.text(f'Analyzing text {i+1} of {len(texts)}...')
                
                # Simple analysis
                polarity, subjectivity = simple_sentiment_analysis(text)
                
                # Advanced analysis
                advanced_scores = advanced_sentiment_analysis(text)
                
                # Determine sentiment category
                if advanced_scores['compound'] >= 0.05:
                    sentiment_category = "Positive"
                elif advanced_scores['compound'] <= -0.05:
                    sentiment_category = "Negative"
                else:
                    sentiment_category = "Neutral"
                
                results.append({
                    'Text': text[:50] + "..." if len(text) > 50 else text,
                    'Full Text': text,
                    'Simple Polarity': round(polarity, 3),
                    'Advanced Compound': round(advanced_scores['compound'], 3),
                    'Sentiment': sentiment_category,
                    'Positive Score': round(advanced_scores['pos'], 3),
                    'Negative Score': round(advanced_scores['neg'], 3)
                })
                progress_bar.progress((i + 1) / len(texts))
            
            status_text.text('Analysis complete!')
            results_df = pd.DataFrame(results)
            
            # Display results
            st.subheader("📈 Results Summary")
            col1, col2, col3 = st.columns(3)
            
            sentiment_counts = results_df['Sentiment'].value_counts()
            with col1:
                st.metric("Positive Texts", sentiment_counts.get('Positive', 0))
            with col2:
                st.metric("Neutral Texts", sentiment_counts.get('Neutral', 0))
            with col3:
                st.metric("Negative Texts", sentiment_counts.get('Negative', 0))
            
            # Detailed results table
            st.subheader("📋 Detailed Results")
            display_df = results_df.drop('Full Text', axis=1)  # Hide full text for display
            st.dataframe(display_df, use_container_width=True)
            
            # Sentiment distribution chart
            st.subheader("📊 Sentiment Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            sentiment_counts.plot(kind='bar', ax=ax, color=['#28a745', '#ffc107', '#dc3545'])
            ax.set_title('Sentiment Distribution')
            ax.set_xlabel('Sentiment')
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=0)
            st.pyplot(fig)
            plt.close()
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="sentiment_analysis_results.csv",
                mime="text/csv"
            )
        else:
            st.warning("⚠️ Please enter some texts to analyze")

with tab3:
    st.header("📊 Sample Data Analysis")
    df = load_sample_data()
    
    st.subheader("📋 Sample Dataset")
    st.dataframe(df, use_container_width=True)
    
    # Analyze sample data
    if st.button("🔍 Analyze Sample Data"):
        with st.spinner("Analyzing sample data..."):
            sample_results = []
            for _, row in df.iterrows():
                scores = advanced_sentiment_analysis(row['text'])
                sample_results.append({
                    'text': row['text'],
                    'actual_sentiment': row['sentiment'],
                    'predicted_compound': scores['compound'],
                    'predicted_sentiment': 'positive' if scores['compound'] >= 0.05 
                                         else 'negative' if scores['compound'] <= -0.05 
                                         else 'neutral'
                })
            
            sample_df = pd.DataFrame(sample_results)
            
            # Accuracy calculation
            correct_predictions = sum(sample_df['actual_sentiment'] == sample_df['predicted_sentiment'])
            accuracy = correct_predictions / len(sample_df) * 100
            
            st.success(f"🎯 Prediction Accuracy: {accuracy:.1f}%")
            st.dataframe(sample_df)
    
    # Word frequency analysis
    st.subheader("🔤 Most Common Words")
    word_freq = create_simple_wordcloud(df['text'].tolist())
    
    if word_freq:
        freq_df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
        
        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            words = [item[0] for item in word_freq[:10]]
            counts = [item[1] for item in word_freq[:10]]
            ax.barh(words, counts)
            ax.set_title('Top 10 Most Frequent Words')
            ax.set_xlabel('Frequency')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.dataframe(freq_df.head(10))

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <h4>🎭 Sentiment Analysis Tool v2.0</h4>
    <p>Built with ❤️ using Streamlit | No External Dependencies Required</p>
    <p><em>Perfect for deployment on any platform!</em></p>
</div>
""", unsafe_allow_html=True)
