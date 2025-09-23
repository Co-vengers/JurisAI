import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    raise SystemExit("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")

GENERATION_MODEL = genai.GenerativeModel("gemini-1.5-flash")

BOOK_PATH = "indian_constitution copy.csv"

CONSTITUTION_DF = None
ARTICLE_EMBEDDINGS = None