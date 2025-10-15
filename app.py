# Import all the required libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
import streamlit as st

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}

# Load the pre-trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('imdb_rnn_model.h5')

model = load_model()

# Function to decode review
def decode_review(encode_review):
    return ' '.join([reverse_word_index.get(i-3, '?') for i in encode_review])

# Function to preprocess the user input
def preprocess_text(text):
    # Tokenize the text
    tokens = text.lower().split()
    # Convert tokens to integers based on the word index
    encoded = [word_index.get(word, 2) + 3 for word in tokens]  # 2 is for unknown words
    # Pad the sequence to ensure it matches the input length of the model
    padded = sequence.pad_sequences([encoded], maxlen=256)
    return padded

# Function to predict sentiment
def predict_sentiment(text):
    # Preprocess the input text
    processed_text = preprocess_text(text)
    # Make prediction
    prediction = model.predict(processed_text, verbose=0)
    # Interpret the prediction
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

# Streamlit app configuration
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    /* Main theme colors */
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1a1a2e;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .main-header h1 {
        color: #ffffff;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
    
    .main-header p {
        color: #e0e0e0;
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Card styling */
    .prediction-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Input styling */
    .stTextArea > div > div > textarea {
        background-color: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.4);
        border-radius: 10px;
        color: #ffffff !important;
        font-size: 1.1rem;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.3);
        background-color: rgba(255, 255, 255, 0.2);
    }
    
    /* Label styling */
    .stTextArea label {
        color: #ffffff !important;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* Placeholder styling */
    .stTextArea > div > div > textarea::placeholder {
        color: #b0b0b0 !important;
        opacity: 0.8;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: #ffffff;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Success/Error message styling */
    .stSuccess {
        background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%);
        border-radius: 10px;
        border: none;
        color: #ffffff !important;
    }
    
    .stSuccess .stAlert {
        color: #ffffff !important;
    }
    
    .stError {
        background: linear-gradient(90deg, #f44336 0%, #d32f2f 100%);
        border-radius: 10px;
        border: none;
        color: #ffffff !important;
    }
    
    .stError .stAlert {
        color: #ffffff !important;
    }
    
    .stInfo {
        background: linear-gradient(90deg, #2196F3 0%, #1976D2 100%);
        border-radius: 10px;
        border: none;
        color: #ffffff !important;
    }
    
    .stInfo .stAlert {
        color: #ffffff !important;
    }
    
    .stWarning {
        background: linear-gradient(90deg, #FF9800 0%, #F57C00 100%);
        border-radius: 10px;
        border: none;
        color: #ffffff !important;
    }
    
    .stWarning .stAlert {
        color: #ffffff !important;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Metric styling */
    .metric-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
        margin: 0;
    }
    
    .metric-label {
        font-size: 1.1rem;
        color: #e0e0e0;
        margin: 0.5rem 0 0 0;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #888;
        font-style: italic;
    }
    
    /* General text visibility fixes */
    .stApp {
        color: #ffffff !important;
    }
    
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #ffffff !important;
    }
    
    .stApp p, .stApp div, .stApp span {
        color: #ffffff !important;
    }
    
    /* Sidebar text visibility */
    .css-1d391kg {
        color: #ffffff !important;
    }
    
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, .css-1d391kg h4, .css-1d391kg h5, .css-1d391kg h6 {
        color: #ffffff !important;
    }
    
    .css-1d391kg p, .css-1d391kg div, .css-1d391kg span {
        color: #ffffff !important;
    }
    
    /* Button text */
    .stButton > button {
        color: #ffffff !important;
    }
    
    /* Spinner text */
    .stSpinner {
        color: #ffffff !important;
    }
    
    /* Help text */
    .stTooltip {
        color: #ffffff !important;
    }
    
    /* Metric text */
    .stMetric {
        color: #ffffff !important;
    }
    
    .stMetric > div > div > div {
        color: #ffffff !important;
    }
    
    /* Write text */
    .stWrite {
        color: #ffffff !important;
    }
    
    /* Markdown text */
    .stMarkdown {
        color: #ffffff !important;
    }
    
    .stMarkdown p, .stMarkdown div, .stMarkdown span {
        color: #ffffff !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🎬 IMDB Sentiment Analysis</h1>
    <p>Advanced AI-powered movie review sentiment prediction</p>
</div>
""", unsafe_allow_html=True)

# Add description in a card
st.markdown("""
<div class="prediction-card">
    <h3 style="color: #667eea; margin-top: 0;">📝 How it works</h3>
    <p style="color: #e0e0e0; font-size: 1.1rem; line-height: 1.6;">
        This app uses a trained Recurrent Neural Network (RNN) with word embeddings to analyze 
        movie reviews and predict whether they express positive or negative sentiment. 
        Enter your movie review below and click 'Classify' to get an instant prediction with confidence score.
    </p>
</div>
""", unsafe_allow_html=True)

# Create two columns for better layout
col1, col2 = st.columns([2, 1])

with col1:
    # User input section
    st.markdown("""
    <div class="prediction-card">
        <h3 style="color: #667eea; margin-top: 0;">✍️ Enter Your Review</h3>
    </div>
    """, unsafe_allow_html=True)
    
    user_input = st.text_area(
        "Enter your movie review:",
        placeholder="Type your movie review here...",
        height=120,
        help="Enter a movie review to analyze its sentiment"
    )

with col2:
    # Example review section
    st.markdown("""
    <div class="prediction-card">
        <h3 style="color: #667eea; margin-top: 0;">💡 Example</h3>
        <p style="color: #e0e0e0; font-style: italic;">
            "The movie was fantastic! I really loved it and would watch it again."
        </p>
    </div>
    """, unsafe_allow_html=True)

# Classification button
st.markdown("<br>", unsafe_allow_html=True)
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    if st.button('🔍 Classify Sentiment', type="primary", use_container_width=True):
        if user_input.strip():
            with st.spinner('🤖 Analyzing sentiment...'):
                # Predict sentiment
                sentiment, confidence = predict_sentiment(user_input)

                # Display results card
                st.markdown("""
                <div class="prediction-card">
                    <h3 style="color: #667eea; margin-top: 0;">📊 Analysis Results</h3>
                </div>
                """, unsafe_allow_html=True)

                # Split results into two columns
                result_col1, result_col2 = st.columns([1, 1])

                with result_col1:
                    # Sentiment display
                    if sentiment == 'Positive':
                        st.markdown("""
                        <div class="metric-container">
                            <div class="metric-value" style="color: #4CAF50;">✅</div>
                            <div class="metric-label">Sentiment</div>
                            <h2 style="color: #4CAF50; margin: 0.5rem 0;">Positive</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="metric-container">
                            <div class="metric-value" style="color: #f44336;">❌</div>
                            <div class="metric-label">Sentiment</div>
                            <h2 style="color: #f44336; margin: 0.5rem 0;">Negative</h2>
                        </div>
                        """, unsafe_allow_html=True)

                with result_col2:
                    # Confidence display
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">{confidence:.1%}</div>
                        <div class="metric-label">Confidence</div>
                        <p style="color: #e0e0e0; margin: 0.5rem 0;">
                            The model is {confidence*100:.1f}% confident
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                # Progress bar
                st.markdown("<br>", unsafe_allow_html=True)
                st.progress(confidence)

                # Additional insights
                if confidence > 0.8:
                    insight = "Very confident prediction! 🎯"
                    insight_color = "#4CAF50"
                elif confidence > 0.6:
                    insight = "Moderately confident prediction 📊"
                    insight_color = "#FF9800"
                else:
                    insight = "Low confidence - review might be ambiguous 🤔"
                    insight_color = "#f44336"

                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; background: rgba(255, 255, 255, 0.1); 
                           border-radius: 10px; margin: 1rem 0;">
                    <p style="color: {insight_color}; font-weight: 600; margin: 0;">{insight}</p>
                </div>
                """, unsafe_allow_html=True)

        else:
            # Handle empty input
            st.markdown("""
            <div style="text-align: center; padding: 1rem; background: rgba(255, 152, 0, 0.2); 
                       border: 1px solid rgba(255, 152, 0, 0.5); border-radius: 10px; margin: 1rem 0;">
                <p style="color: #FF9800; font-weight: 600; margin: 0;">⚠️ Please enter a movie review before classifying.</p>
            </div>
            """, unsafe_allow_html=True)


# Sidebar with additional information
with st.sidebar:
    st.markdown("""
    <div style="background: rgba(255, 255, 255, 0.15); padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem; border: 1px solid rgba(255, 255, 255, 0.2);">
        <h3 style="color: #667eea; margin-top: 0; font-weight: 700;">🔧 Model Info</h3>
        <p style="color: #ffffff; font-size: 0.9rem; line-height: 1.6; margin: 0;">
            <strong style="color: #667eea;">Architecture:</strong> Simple RNN<br>
            <strong style="color: #667eea;">Embedding Dim:</strong> 128<br>
            <strong style="color: #667eea;">Vocabulary:</strong> 88,585 words<br>
            <strong style="color: #667eea;">Accuracy:</strong> ~85%
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(255, 255, 255, 0.15); padding: 1.5rem; border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.2);">
        <h3 style="color: #667eea; margin-top: 0; font-weight: 700;">💡 Tips</h3>
        <ul style="color: #ffffff; font-size: 0.9rem; padding-left: 1.2rem; line-height: 1.6; margin: 0;">
            <li style="margin-bottom: 0.5rem;">Write detailed reviews for better accuracy</li>
            <li style="margin-bottom: 0.5rem;">Include emotional words (amazing, terrible, etc.)</li>
            <li style="margin-bottom: 0.5rem;">Avoid sarcasm or complex negations</li>
            <li style="margin-bottom: 0.5rem;">Reviews with 20+ words work best</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    <p>🚀 Powered by TensorFlow & Streamlit | Made with ❤️ for the ML community</p>
</div>
""", unsafe_allow_html=True)

