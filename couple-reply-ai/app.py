# app.py
import streamlit as st
from utils.emotion_detector import detect_emotion
from utils.reply import generate_reply, ReplyQuality
from utils.rag_memory import add_to_memory

st.title("💬 Couple Message Replier")

partner_msg = st.text_area("Partner's Message")

if partner_msg:
    emotion = detect_emotion(partner_msg)
    st.markdown(f"**Detected Emotion:** `{emotion}`")

    intent = st.selectbox("Your Intent", ["Calm Them", "Be Funny", "Be Sarcastic", "Apologize", "Fight Back", "Show Empathy", "Be Supportive", "Change the Topic"])
    
    quality = st.select_slider(
        "Reply Quality",
        options=["best", "good", "neutral", "bad"],
        value="best",
        help="Choose the quality level of the generated response"
    )
    
    if st.button("Generate Reply"):
        reply = generate_reply(partner_msg, emotion, intent, quality)
        st.success("Generated Reply:")
        st.write(reply)

        user_input = st.text_input("Your Reply (to store in memory):", value=reply)
        if st.button("Save this conversation"):
            add_to_memory(user_input, partner_msg)
            st.success("Memory updated!")