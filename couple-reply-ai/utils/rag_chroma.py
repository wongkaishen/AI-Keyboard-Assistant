import os
from chromadb import PersistentClient
from google.generativeai import embed_content
from dotenv import load_dotenv

load_dotenv()

# Initialize Chroma client with PersistentClient instead of Client+Settings
chroma_client = PersistentClient(path="./chroma_storage")
COLLECTION_NAME = "partner_replies"

# Get or create collection
try:
    collection = chroma_client.get_collection(COLLECTION_NAME)
except:
    collection = chroma_client.create_collection(COLLECTION_NAME)

def get_embedding(text: str):
    """Use Gemini to get a 768-dimensional embedding"""
    result = embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    return result["embedding"]

def search_similar(query: str, top_k: int = 3):
    """Search Chroma for top_k similar partner messages"""
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results["documents"][0] if results["documents"] else []
