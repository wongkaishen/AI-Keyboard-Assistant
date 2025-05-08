# utils/rag_memory.py

message_history = []

def add_to_memory(user_msg, partner_msg):
    message_history.append({
        "user": user_msg,
        "partner": partner_msg
    })

def retrieve_relevant_memory(current_msg, top_k=3):
    """
    Simple keyword-based retrieval. Later upgrade to embeddings.
    """
    relevant = sorted(
        message_history,
        key=lambda m: int(current_msg in m["partner"] or current_msg in m["user"]),
        reverse=True
    )
    return relevant[:top_k]
