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
        pass
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

def load_and_clean_data(file_path, encoding='utf-8'):
    try:
        data = pd.read_csv(file_path, encoding=encoding)
    except UnicodeDecodeError:
        data = pd.read_csv(file_path, encoding='latin-1')

    def clean_text(text):
        if isinstance(text, str):
            text = ''.join(char for char in text if char.isprintable())
            text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
            text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
        return text

    data.columns = data.columns.str.replace('ï»¿', '').str.replace('Ã', '').str.strip()
    for col in data.columns:
        data[col] = data[col].apply(clean_text)

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

def get_availability_status(row, availability_data):
    if availability_data is None:
        return "Unknown"
        
    lawyer = availability_data[
        (availability_data['First Name'].str.strip() == row['First Name'].strip()) &
        (availability_data['Last Name'].str.strip() == row['Last Name'].strip())
    ]
    
    if lawyer.empty:
        return "Unknown"
        
    can_take_work = lawyer['Do you have capacity to take on new work?'].iloc[0]
    
    if can_take_work == 'No':
        return "Not Available"
    elif can_take_work == 'Maybe':
        return "Limited Availability"
    
    days_per_week = lawyer['What is your capacity to take on new work for the forseeable future? Days per week'].iloc[0]
    hours_per_month = lawyer['What is your capacity to take on new work for the foreseeable future? Hours per month'].iloc[0]
    
    try:
        max_days = max(int(day) for day in re.findall(r'\d+', str(days_per_week)))
    except ValueError:
        max_days = 0
    
    if max_days >= 4 and 'More than 80 hours' in hours_per_month:
        return "High Availability"
    elif max_days >= 2:
        return "Moderate Availability"
    else:
        return "Low Availability"

def expand_query(query):
    expanded_query = []
    for word, tag in nltk.pos_tag(nltk.word_tokenize(query)):
        synsets = wordnet.synsets(word)
        if synsets:
            synonyms = set()
            for synset in synsets:
                synonyms.update(lemma.name().replace('_', ' ') for lemma in synset.lemmas())
            expanded_query.extend(list(synonyms)[:3])
        expanded_query.append(word)
    return ' '.join(expanded_query)

def normalize_query(query):
    query = re.sub(r'[^\w\s]', '', query)
    return query.lower()

def preprocess_query(query):
    tokens = word_tokenize(query)
    tagged = pos_tag(tokens)
    keywords = [word.lower() for word, pos in tagged if pos in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS']]
    stop_words = ['lawyer', 'best', 'top', 'find', 'me', 'give', 'who']
    keywords = [word for word in keywords if word not in stop_words]
    return keywords

def keyword_search(data, query_keywords):
    search_columns = ['Area of Practise + Add Info', 'Industry Experience', 'Expert', 'Lawyer Bio Info']
    
    def contains_any_term(text):
        if not isinstance(text, str):
            return False
        text_lower = text.lower()
        return any(term in text_lower for term in query_keywords)
    
    masks = [data[col].apply(contains_any_term) for col in search_columns]
    final_mask = masks[0]
    for mask in masks[1:]:
        final_mask |= mask
    
    return data[final_mask]

def calculate_relevance_score(text, query_keywords):
    if not isinstance(text, str):
        return 0
    text_lower = text.lower()
    return sum(text_lower.count(keyword) for keyword in query_keywords)

@st.cache_resource
def create_weighted_vector_db(data):
    def weighted_text(row):
        return ' '.join([
            str(row['First Name']),
            str(row['Last Name']),
            str(row['Level/Title']),
            str(row['Call']),
            str(row['Jurisdiction']),
            str(row['Location']),
            str(row['Area of Practise + Add Info']),
            str(row['Industry Experience']),
            str(row['Languages']),
            str(row['Previous In-House Companies']),
            str(row['Previous Companies/Firms']),
            str(row['Education']),
            str(row['Awards/Recognition']),
            str(row['City of Residence']),
            str(row['Notable Items/Personal Details']),
            str(row['Expert']),
            str(row['Lawyer Bio Info'])
        ])

    combined_text = data.apply(weighted_text, axis=1)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(combined_text)
    X_normalized = normalize(X, norm='l2', axis=1, copy=False)
    
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(np.ascontiguousarray(X_normalized.toarray()))
    return index, vectorizer

def query_claude_with_data(question, matters_data, matters_index, matters_vectorizer):
    availability_data = load_availability_data('Caravel Law Availability - October 18th, 2024.csv')
    query_keywords = preprocess_query(question)
    keyword_results = keyword_search(matters_data, query_keywords)
    
    relevant_data = keyword_results.copy() if not keyword_results.empty else matters_data.copy()

    relevant_data['keyword_score'] = relevant_data.apply(
        lambda row: calculate_relevance_score(' '.join(row.astype(str)), query_keywords), axis=1
    )

    question_vec = matters_vectorizer.transform([' '.join(query_keywords)])
    k = min(len(relevant_data), 10)
    D, I = matters_index.search(normalize(question_vec).toarray(), k=k)
    
    relevant_data['semantic_score'] = 0
    valid_indices = [idx for idx in I[0] if idx < len(relevant_data)]
    if valid_indices:
        scores = 1 / (1 + D[0][:len(valid_indices)])
        relevant_data.iloc[valid_indices, relevant_data.columns.get_loc('semantic_score')] = scores

    relevant_data['relevance_score'] = (relevant_data['keyword_score'] * 0.7) + (relevant_data['semantic_score'] * 0.3)

    availability_weights = {
        "High Availability": 1.0,
        "Moderate Availability": 0.8,
        "Low Availability": 0.6,
        "Limited Availability": 0.4,
        "Not Available": 0.1,
        "Unknown": 0.5
    }
    relevant_data['availability_weight'] = relevant_data['availability_status'].map(availability_weights)
    relevant_data['final_score'] = relevant_data['relevance_score'] * relevant_data['availability_weight']

    top_relevant_data = relevant_data.sort_values('final_score', ascending=False)
    primary_info = top_relevant_data[['First Name', 'Last Name', 'Level/Title', 'Call', 'Jurisdiction', 'Location', 
                                    'Area of Practise + Add Info', 'Industry Experience', 'Education', 'availability_status', 
                                    'Lawyer Bio Info']].drop_duplicates(subset=['First Name', 'Last Name'])

    messages = [
        {"role": "system", "content": "You are an expert legal consultant..."},
        {"role": "user", "content": f"Core query keywords: {', '.join(query_keywords)}\nOriginal question: {question}\n\nTop Lawyers Information:\n{primary_info.to_string(index=False)}"}
    ]

    claude_response = call_claude(messages)
    if not claude_response:
        return
    log_query_and_result(question, claude_response)
    st.write("### Claude's Recommendation:")
    st.write(claude_response)

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

