# utils/embedding_utils.py
import os
from dotenv import load_dotenv
import google.generativeai as genai
import numpy as np

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)  # or use env variable

def get_embedding(text):
    model = genai.GenerativeModel("embedding-001")
    res = model.embed_content(content=text, task_type="retrieval_document")
    return np.array(res['embedding'])
