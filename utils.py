import numpy as np
import pandas as pd
import google.generativeai as genai
from config import BOOK_PATH, GENERATION_MODEL
from functools import lru_cache
import json

class Utils:
    def preprocess_and_embed_data(self):
        global CONSTITUTION_DF, ARTICLE_EMBEDDINGS
        try:
            df = pd.read_csv(BOOK_PATH)
            df['Text'] = df['Article Heading'] + ". " + df['Article Description']
            df.dropna(subset=['Text'], inplace=True)
            CONSTITUTION_DF = df
            print(f"Successfully loaded and processed {len(CONSTITUTION_DF)} articles.")

            print("Generating embeddings for all articles. This may take a moment...")

            response = genai.embed_content(
                model="models/text-embedding-004",  # The model for embedding
                content=CONSTITUTION_DF['Text'].tolist(),
                task_type="retrieval_document"
            )
            ARTICLE_EMBEDDINGS = np.array(response['embedding'])
            print("Embeddings generated successfully.")

        except FileNotFoundError:
            raise SystemExit(f"Error: The file was not found at {BOOK_PATH}")
        except KeyError:
            raise SystemExit("Error: CSV must contain 'Article Heading' and 'Article Description' columns.")
        except Exception as e:
            raise SystemExit(f"An error occurred during data preprocessing: {e}")

    def find_relevant_articles(self, query, top_k=5):
        query_embedding_response = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="retrieval_query"
        )
        query_embedding = np.array(query_embedding_response['embedding'])

        dot_products = np.dot(ARTICLE_EMBEDDINGS, query_embedding)
        norms = np.linalg.norm(ARTICLE_EMBEDDINGS, axis=1) * np.linalg.norm(query_embedding)
        similarities = dot_products / norms

        top_indices = np.argsort(similarities)[-top_k:][::-1]
        relevant_context = "\n\n".join(CONSTITUTION_DF.iloc[top_indices]['Text'].tolist())
        return relevant_context

    @lru_cache(maxsize=128)
    def get_chatbot_response(self, user_input):
        relevant_context = self.find_relevant_articles(user_input)
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
