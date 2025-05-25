**Sentiment Analysis Tool**:

* A simple web application for analyzing text sentiment built with Streamlit. No external APIs or complex dependencies required.

# Features:

* **Single Text Analysis** - Analyze individual texts with sentiment scores
* **Batch Processing** - Process multiple texts at once
* **Visual Results** - Charts and graphs showing sentiment distribution
* **Export Data** - Download results as CSV
* **Sample Data** - Pre-loaded examples to test the tool

**Live Demo**:


# Installation

* Clone the repository
bashgit clone https://github.com/yourusername/sentiment-analysis-tool.git
cd sentiment-analysis-tool

* Install requirements
bashpip install -r requirements.txt

* Run the app
bashstreamlit run app.py

Open your browser to http://localhost:8501

# Requirements

*streamlit>=1.28.0

* pandas>=1.5.0
  
* matplotlib>=3.6.0
  
# How It Works

The tool uses rule-based sentiment analysis with:

* Positive/negative word detection
  
* Intensifier handling (very, extremely, really)

* Negation detection (not, never, no)

* Context-aware scoring

# Usage Examples

* Single Text

* Input: "I love this product!"

* Output: Positive (😊) - Score: 0.65

* Batch Processing

* Input: Multiple texts (one per line)

* Output: CSV with sentiment scores for each text
  
# Deployment

Streamlit Cloud

# Push code to GitHub

Connect to share.streamlit.io
Deploy automatically

# Other Platforms
Works on Heroku, Railway, Render, and other hosting platforms.

# File Structure

sentiment-analysis-tool/
├── app.py              # Main application
├── requirements.txt    # Dependencies
└── README.md          # This file
Contributing

# Fork the repository

Create a feature branch
Make your changes
Submit a pull request

# License

MIT License - feel free to use and modify.

Built with ❤️ using Streamlit
