
import streamlit as st
from main import run_agent
import os

# Load environment variables (if not loaded in main)
from dotenv import load_dotenv
load_dotenv()

# Streamlit page configuration
st.set_page_config(page_title="Multi-Agent AI ChatBot", page_icon=":robot_face:")

st.title("Multi-Agent AI ChatBot 🤖")

# Initialize session state variables if not already present
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    # Add a default welcome message
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": "Welcome to the Multi-Agent AI ChatBot! 🤖 How can I assist you today?"
    })

# Placeholder for chat display
chat_placeholder = st.empty()

# Function to display chat history
def display_chat_history():
    with chat_placeholder.container():
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                col1, col2 = st.columns([1, 5])
                with col2:
                    st.chat_message("user").write(message["content"])
            else:
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.chat_message("assistant").write(message["content"])

# Display the current chat history
display_chat_history()

# Input widget for user queries
user_input = st.chat_input("Please enter your query here!")

if user_input:
    # Append user message to history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    display_chat_history()

    # Generate response
    with st.spinner("Processing..."):
        assistant_response = run_agent(user_input)

    # Append assistant response to history
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
    display_chat_history()
