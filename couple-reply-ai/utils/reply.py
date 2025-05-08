import os
from dotenv import load_dotenv
from typing import Literal

import google.generativeai as genai
from utils.rag_faiss import search_similar
from utils.rag_memory import retrieve_relevant_memory
from utils.rag_chroma import search_similar

# Configure the Gemini API
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Define valid reply quality levels
ReplyQuality = Literal["best", "good", "neutral", "bad"]

def generate_reply(partner_msg: str, emotion: str, intent: str, quality: ReplyQuality = "best"):
    # Initialize the model
    past_contexts = search_similar(partner_msg)
    memory_snippets = retrieve_relevant_memory(partner_msg)
    memory_str = "\n".join(
        [f"Partner: {m['partner']}\nUser: {m['user']}" for m in memory_snippets]
        + [f"Context: {c}" for c in past_contexts]
    ) or "No history available."
    model = genai.GenerativeModel('gemini-1.5-flash-002')
    
    # Define intention instructions
    intent_instructions = {
        "Calm Them": "Create a soothing, reassuring message that helps de-escalate their emotions.",
        "Be Funny": "Respond with humor and lightheartedness to brighten their mood.",
        "Be Sarcastic": "Use witty sarcasm (but not cruel) to respond to their message.",
        "Apologize": "Offer a sincere apology that acknowledges their feelings.",
        "Fight Back": "Defend your position firmly but without being unnecessarily hurtful.",
        "Show Empathy": "Demonstrate that you understand and share their feelings.",
        "Be Supportive": "Offer encouragement and reassurance to show you're on their side.",
        "Change the Topic": "Gently shift the conversation to a different, more positive subject."
    }
    
    # Define quality level instructions
    quality_instructions = {
        "best": "Make this the most thoughtful, empathetic and perfect response possible.",
        "good": "Create a solid, appropriate response that addresses the message well.",
        "neutral": "Craft a basic, somewhat generic response that's acceptable but not exceptional.",
        "bad": "Write a response that's somewhat careless, with minimal effort and attention to the partner's needs."
    }
    
    system_prompt = """You are an empathetic AI assistant specifically designed to help craft responses in 
    conversations between partners. Respond directly with the suggested message, without any explanations 
    or additional text."""
    {memory_str}
    user_prompt = f"""The partner said (with {emotion} tone): "{partner_msg}"

Intention: {intent_instructions[intent]}

Quality level: {quality_instructions[quality]}

Based on these guidelines, craft a message that responds to their communication."""
    
    combined_prompt = f"{system_prompt}\n\n{user_prompt}"
    
    # Generate response using Gemini
    response = model.generate_content(combined_prompt)
    
    return response.text.strip()