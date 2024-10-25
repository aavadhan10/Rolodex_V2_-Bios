import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from anthropic import Anthropic
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

# Download required NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt', quiet=True)

def log_query_and_result(query, result):
    """Log queries and results to a CSV file"""
    log_file = "query_log.csv"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        # Create the file with headers if it doesn't exist
        if not os.path.exists(log_file):
            with open(log_file, "w", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Query", "Result"])
        
        # Append the new query and result
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
    claude_api_key = st.secrets["CLAUDE_API_KEY"]
    if not claude_api_key:
        st.error("Anthropic API key not found. Please check your Streamlit secrets configuration.")
        st.stop()
    return Anthropic(api_key=claude_api_key)

client = init_anthropic_client()

def load_and_clean_data(file_path, encoding='utf-8'):
    try:
        data = pd.read_csv(file_path, encoding=encoding)
    except UnicodeDecodeError:
        data = pd.read_csv(file_path, encoding='latin-1')

    def clean_text(text):
        if isinstance(text, str):
            text = ''.join(char for char in text if char.isprintable())
            text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
            text = text.replace('Ã¢ÂÂ', "'").replace('Ã¢ÂÂ¨', ", ")
            text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
        return text

    data.columns = data.columns.str.replace('ï»¿', '').str.replace('Ã', '').str.strip()
    for col in data.columns:
        data[col] = data[col].apply(clean_text)
    
    column_mappings = {
        'First Name': 'First Name',
        'Last Name': 'Last Name',
        'Level/Title': 'Level/Title',
        'Call': 'Call',
        'Jurisdiction': 'Jurisdiction',
        'Location': 'Location',
        'Area of Practise + Add Info': 'Area of Practise + Add Info',
        'Industry Experience': 'Industry Experience',
        'Languages': 'Languages',
        'Expert': 'Expert',
        'Previous In-House Companies': 'Previous In-House Companies',
        'Previous Companies/Firms': 'Previous Companies/Firms',
        'Education': 'Education',
        'Awards/Recognition': 'Awards/Recognition',
        'City of Residence': 'City of Residence',
        'Notable Items/Personal Details': 'Notable Items/Personal Details',
        'Daily/Fractional Engagements': 'Daily/Fractional Engagements',
        'Monthly Engagements (hours per month)': 'Monthly Engagements (hours per month)',
        'Lawyer Bio Info': 'Lawyer Bio Info'
    }
    
    data = data.rename(columns=column_mappings)
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    return data

def load_availability_data(file_path):
    availability_data = pd.read_csv(file_path)
    availability_data.columns = [col.strip() for col in availability_data.columns]
    availability_data['What is your name?'] = availability_data['What is your name?'].str.strip()
    availability_data['First Name'] = ''
    availability_data['Last Name'] = ''
    
    for idx, row in availability_data.iterrows():
        name_parts = str(row['What is your name?']).split()
        if len(name_parts) >= 2:
            availability_data.at[idx, 'First Name'] = name_parts[0]
            availability_data.at[idx, 'Last Name'] = ' '.join(name_parts[1:])
        elif len(name_parts) == 1:
            availability_data.at[idx, 'First Name'] = name_parts[0]
            availability_data.at[idx, 'Last Name'] = ''
    
    availability_data['First Name'] = availability_data['First Name'].str.strip()
    availability_data['Last Name'] = availability_data['Last Name'].str.strip()
    return availability_data

def display_available_lawyers():
    """Display all available lawyers and their capacity"""
    availability_data = load_availability_data('Caravel Law Availability - October 18th, 2024.csv')
    matters_data = load_and_clean_data('BD_Caravel.csv')
    available_lawyers = availability_data[availability_data['Do you have capacity to take on new work?'].isin(['Yes', 'Maybe'])]
    
    st.write("### Currently Available Lawyers")
    for _, lawyer in available_lawyers.iterrows():
        name = f"{lawyer['First Name']} {lawyer['Last Name']}"
        lawyer_info = matters_data[
            (matters_data['First Name'] == lawyer['First Name']) & 
            (matters_data['Last Name'] == lawyer['Last Name'])
        ]
        
        practice_areas = lawyer_info['Area of Practise + Add Info'].iloc[0] if not lawyer_info.empty else "Information not available"
        
        with st.expander(f"🧑‍⚖️ {name} - {'Ready for New Work' if lawyer['Do you have capacity to take on new work?'] == 'Yes' else 'Limited Availability'}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Availability Details:**")
                st.write(f"• Days per week: {lawyer['What is your capacity to take on new work for the forseeable future? Days per week']}")
                st.write(f"• Hours per month: {lawyer['What is your capacity to take on new work for the foreseeable future? Hours per month']}")
                st.write(f"• Preferred engagement types: {lawyer['What type of engagement would you like to consider?']}")
            
            with col2:
                st.write("**Practice Areas:**")
                st.write(practice_areas)
            
            notes = lawyer['Do you have any comments or instructions you should let us know about that may impact your short/long-term availability? For instance, are you going on vacation (please provide exact dates)?']
            if pd.notna(notes) and notes.lower() not in ['no', 'n/a', 'none', 'nil']:
                st.write("**Availability Notes:**")
                st.write(notes)

# Streamlit app layout
st.title("Rolodex Caravel Version 2 Lawyer Bio 👨‍⚖️ Utilizing Claude 3.5")
st.write("Ask questions about the skill-matched lawyers for your specific legal needs and their availability:")

# Add tabs for different views
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
        matters_data = load_and_clean_data('BD_Caravel.csv')
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

