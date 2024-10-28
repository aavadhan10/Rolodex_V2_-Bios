import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from anthropic import Anthropic, APIError
import re
import unicodedata
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import datetime
import os
import csv
from collections import Counter
import base64

def safe_nltk_download(package):
    """Safely download NLTK data packages"""
    try:
        nltk.download(package, quiet=True)
    except FileExistsError:
        pass  # Directory already exists, which is fine
    except Exception as e:
        st.warning(f"Warning: Could not download NLTK package {package}. Error: {str(e)}")

# Initialize NLTK downloads safely
try:
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    safe_nltk_download('wordnet')
    safe_nltk_download('averaged_perceptron_tagger')
    safe_nltk_download('punkt')
except Exception as e:
    st.warning(f"Warning: Issue with NLTK setup. Some features may be limited. Error: {str(e)}")

def log_query_and_result(query, result):
    """Log queries and results to a CSV file"""
    log_file = "query_log.csv"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        if not os.path.exists(log_file):
            with open(log_file, "w", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Query", "Result"])
        
        with open(log_file, "a", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, query, result])
    except Exception as e:
        st.warning(f"Unable to log query: {str(e)}")

def get_most_asked_queries(n=10):
    """Get the most frequently asked queries from the log"""
    if not os.path.exists("query_log.csv"):
        return pd.DataFrame(columns=["Query", "Count", "Last Result"])
    
    try:
        df = pd.read_csv("query_log.csv")
        query_counts = Counter(df["Query"])
        most_common = query_counts.most_common(n)
        
        results = []
        for query, count in most_common:
            last_result = df[df["Query"] == query].iloc[-1]["Result"]
            results.append({
                "Query": query,
                "Count": count,
                "Last Result": last_result
            })
        
        return pd.DataFrame(results)
    except Exception as e:
        st.warning(f"Error getting query statistics: {str(e)}")
        return pd.DataFrame(columns=["Query", "Count", "Last Result"])

def get_csv_download_link(df, filename="most_asked_queries.csv"):
    """Create a download link for CSV data"""
    try:
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV file</a>'
        return href
    except Exception as e:
        st.warning(f"Error creating download link: {str(e)}")
        return ""
        
def init_anthropic_client():
    """Initialize Anthropic client with API key"""
    try:
        claude_api_key = st.secrets["CLAUDE_API_KEY"]
        if not claude_api_key:
            st.error("Anthropic API key not found. Please check your Streamlit secrets configuration.")
            st.stop()
        return Anthropic(api_key=claude_api_key)
    except Exception as e:
        st.error(f"Error initializing Anthropic client: {e}")
        st.stop()

# Create the client instance
client = init_anthropic_client()

def call_claude(messages):
    """Call Claude API using Messages format"""
    try:
        system_message = messages[0]['content'] if messages[0]['role'] == 'system' else ""
        user_message = next(msg['content'] for msg in messages if msg['role'] == 'user')
        
        # Using the Completion API format required for Claude 3.5
        response = client.completions.create(
            model="claude-3-sonnet-20240229",
            prompt=f"{system_message}\n{user_message}",
            temperature=0.7,
            max_tokens_to_sample=500
        )
        
        return response.completion
    except APIError as e:
        st.error(f"API Error: {e}")
        return None
    except Exception as e:
        st.error(f"Error calling Claude: {e}")
        return None

# Remaining functions (data loading, availability status, etc.) go here unchanged
# Load data, process availability, search and recommend lawyers, etc.

# Streamlit app layout
st.title("Rolodex Caravel Version 2 Lawyer Bio 👨‍⚖️ Utilizing Claude 3.5")
st.write("Ask questions about the skill-matched lawyers for your specific legal needs and their availability:")

tab1, tab2 = st.tabs(["Search Lawyers by Skillset", "View Available Lawyers Week (Updated 10/23/24)"])

with tab1:
    default_questions = {
        "Which lawyers have the most experience with intellectual property?": "intellectual property",
        "Can you recommend a lawyer specializing in employment law?": "employment law",
        "Who are the best lawyers for financial cases?": "financial law",
        "Which lawyer should I contact for real estate matters?": "real estate"
    }

    user_input = st.text_input("Type your question:", placeholder="e.g., 'Who are the top lawyers for corporate law?'")

    for question, _ in default_questions.items():
        if st.button(question):
            user_input = question
            break

    if user_input:
        progress_bar = st.progress(0)
        progress_bar.progress(10)
        matters_data = load_and_clean_data('Updated_Lawyer_Bio_Data.csv')
        if not matters_data.empty:
            progress_bar.progress(50)
            matters_index, matters_vectorizer = create_weighted_vector_db(matters_data)
            progress_bar.progress(90)
            query_claude_with_data(user_input, matters_data, matters_index, matters_vectorizer)
            progress_bar.progress(100)
        else:
            st.error("Failed to load data.")
        progress_bar.empty()

with tab2:
    if st.button("Show Available Lawyers"):
        display_available_lawyers()

# Admin section (hidden by default)
if st.experimental_get_query_params().get("admin", [""])[0].lower() == "true":
    st.write("---")
    st.write("## Admin Section")
    if st.button("Download Most Asked Queries and Results"):
        df_most_asked = get_most_asked_queries()
        st.write(df_most_asked)
        st.markdown(get_csv_download_link(df_most_asked), unsafe_allow_html=True)




