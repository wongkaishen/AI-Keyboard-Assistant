import os
from chromadb import PersistentClient
from dotenv import load_dotenv
from google.generativeai import embed_content
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)


# Use PersistentClient instead of Client+Settings
chroma_client = PersistentClient(path="./chroma_storage")

COLLECTION_NAME = "partner_replies"

# Create or load the collection
try:
    collection = chroma_client.get_collection(COLLECTION_NAME)
except:
    collection = chroma_client.create_collection(COLLECTION_NAME)

# Your sample messages (these can be real messages between couples)
examples = [
    "I can't believe you forgot again. Do I even matter to you?",
    "Hey, I had a rough day, just need a hug.",
    "You never listen to me.",
    "Haha that meme you sent earlier was hilarious 😂",
    "Can we talk later? I'm not in the mood right now.",
    "Why didn't you text me back last night?",
    "I really appreciate everything you’ve done for me this week.",
    "Stop acting like everything is okay when it’s not.",
    "You're the best thing that's happened to me."
]

def get_embedding(text: str):
    result = embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    return result["embedding"]

# Add to Chroma with Gemini embeddings
def insert_examples(texts):
    for idx, text in enumerate(texts):
        embedding = get_embedding(text)
        collection.add(
            ids=[f"msg-{idx}"],
            documents=[text],
            embeddings=[embedding]
        )
    print(f"{len(texts)} messages inserted into ChromaDB.")

if __name__ == "__main__":
    insert_examples(examples)
