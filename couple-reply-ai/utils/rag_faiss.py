# utils/rag_faiss.py

import faiss
import numpy as np
from utils.embedding_utils import get_embedding

message_texts = []
message_vectors = []
index = None

def add_message(text):
    global index
    vector = get_embedding(text).astype("float32")
    message_texts.append(text)
    message_vectors.append(vector)

    if index is None:
        index = faiss.IndexFlatL2(len(vector))
    index.add(np.array([vector]))

def search_similar(query, k=3):
    if index is None:
        return []

    qvec = get_embedding(query).astype("float32")
    distances, ids = index.search(np.array([qvec]), k)
    return [message_texts[i] for i in ids[0] if i < len(message_texts)]
