import os
import json
from functools import lru_cache

import google.generativeai as genai
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, redirect, render_template, request, session, url_for

# --- 1. INITIAL SETUP & CONFIGURATION ---

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API with your key
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    raise SystemExit("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")

# Initialize the Flask application
app = Flask(__name__)
# A secret key is required for Flask session management
app.secret_key = os.urandom(24)

# --- 2. MODEL AND DATA LOADING ---

# Initialize the Gemini model for text generation
GENERATION_MODEL = genai.GenerativeModel("gemini-1.5-flash")

# Define the path to your constitution data CSV file
BOOK_PATH = "indian_constitution copy.csv"

# Global variables to hold the pre-processed data (DataFrame and Embeddings)
CONSTITUTION_DF = None
ARTICLE_EMBEDDINGS = None

def preprocess_and_embed_data():
    # Reads the constitution data, preprocesses it, and generates embeddings.
    # This function runs only once when the application starts.
    global CONSTITUTION_DF, ARTICLE_EMBEDDINGS
    try:
        df = pd.read_csv(BOOK_PATH)
        df['Text'] = df['Article Heading'] + ". " + df['Article Description']
        df.dropna(subset=['Text'], inplace=True)
        CONSTITUTION_DF = df
        print(f"✅ Successfully loaded and processed {len(CONSTITUTION_DF)} articles.")

        print("⏳ Generating embeddings for all articles. This may take a moment...")
        # --- THIS IS THE CORRECTED PART ---
        # Use the top-level genai.embed_content function
        response = genai.embed_content(
            model="models/text-embedding-004",  # The model for embedding
            content=CONSTITUTION_DF['Text'].tolist(),
            task_type="retrieval_document"
        )
        ARTICLE_EMBEDDINGS = np.array(response['embedding'])
        print("✅ Embeddings generated successfully.")

    except FileNotFoundError:
        raise SystemExit(f"Error: The file was not found at {BOOK_PATH}")
    except KeyError:
        raise SystemExit("Error: CSV must contain 'Article Heading' and 'Article Description' columns.")
    except Exception as e:
        raise SystemExit(f"An error occurred during data preprocessing: {e}")

# --- 3. CORE CHATBOT LOGIC (RETRIEVAL-AUGMENTED GENERATION) ---

def find_relevant_articles(query, top_k=5):
    # Finds the most relevant articles to a user's query using vector similarity.
    # --- THIS IS THE CORRECTED PART ---
    # Generate an embedding for the user's query using the correct function
    query_embedding_response = genai.embed_content(
        model="models/text-embedding-004",
        content=query,
        task_type="retrieval_query"
    )
    query_embedding = np.array(query_embedding_response['embedding'])

    # Calculate cosine similarity
    dot_products = np.dot(ARTICLE_EMBEDDINGS, query_embedding)
    norms = np.linalg.norm(ARTICLE_EMBEDDINGS, axis=1) * np.linalg.norm(query_embedding)
    similarities = dot_products / norms

    # Get the top_k most similar articles
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    relevant_context = "\n\n".join(CONSTITUTION_DF.iloc[top_indices]['Text'].tolist())
    return relevant_context

@lru_cache(maxsize=128)
def get_chatbot_response(user_input):
    #Generates a chatbot response using RAG.
    relevant_context = find_relevant_articles(user_input)
    prompt = f"""
    You are an expert on the Indian Constitution. Answer the user's question based *only* on the following context provided from the constitution.
    If the answer is not found in the context, clearly state that the information is not available in the provided articles. Do not make up information.

    **Context from the Constitution:**
    ---
    {relevant_context}
    ---

    **User's Question:**
    {user_input}

    **Answer:**
    """
    try:
        response = GENERATION_MODEL.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error during API call: {e}")
        return "Sorry, I encountered an error while generating a response. Please try again."

# --- 4. FLASK WEB ROUTES ---

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'history' not in session:
        session['history'] = []

    if request.method == 'POST':
        user_query = request.form.get('user_query')
        if user_query:
            bot_response = get_chatbot_response(user_query)
            session['history'].append({'question': user_query, 'answer': bot_response})
            session.modified = True
        return redirect(url_for('index'))

    return render_template('index.html', conversation_history=session['history'])

@app.route('/clear_history', methods=['POST'])
def clear_history():
    session.pop('history', None)
    return redirect(url_for('index'))

# --- 5. APPLICATION STARTUP ---
if __name__ == '__main__':
    preprocess_and_embed_data()
    app.run(debug=True)