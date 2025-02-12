import os
import streamlit as st
import logging
from blueberry import (
    upload_pdf,
    load_pdf,
    split_text,
    index_docs,
    retrieve_docs,
    answer_question,
    check_openai_api_key,
    initialize_pinecone,
    initialize_llm,
    initialize_prompt,
    _text_wrap
)

# Set page config
st.set_page_config(
    page_title="DeepSeek R1 RAG Basic",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize message history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Streamlit's open-source Python library!"}
    ]

# Add a heading for the app
st.header("Chat with the Streamlit docs 💬 📚")

# Custom CSS with improved styling
st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 1rem;
    }
    
    /* Header styling */
    .stTitle {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 2rem !important;
        color: #1E3A8A;
        text-align: center;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    
    /* Upload section in sidebar */
    .uploadSection {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin-bottom: 1.5rem;
        border: 2px dashed #CBD5E1;
    }
    
    /* Question input styling */
    .stTextInput > div > div > input {
        border-radius: 0.5rem;
        border: 2px solid #E2E8F0;
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    /* Thinking process section */
    .thinking-process {
        background-color: #F8FAFC;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Answer section */
    .answer-section {
        background-color: #FFFFFF;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin-top: 1rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    /* Alerts and messages */
    .stAlert {
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 0.5rem;
    }
    
    /* Success message */
    .success-message {
        background-color: #ECFDF5;
        color: #065F46;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #A7F3D0;
        margin: 1rem 0;
    }
    
    /* Spinner */
    .stSpinner {
        text-align: center;
        padding: 1rem;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #F8FAFC;
        border-radius: 0.5rem;
        border: 1px solid #E2E8F0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar header */
    .sidebar-header {
        text-align: center;
        padding-bottom: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Sidebar title */
    .sidebar-title {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #1E3A8A;
        margin-bottom: 0.5rem !important;
    }
    
    .sidebar-subtitle {
        font-size: 1rem;
        color: #4B5563;
    }
    
    /* Main content area */
    .main-content {
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    </style>
""", unsafe_allow_html=True)

def extract_thinking_and_answer(response: str):
    # Placeholder function to extract thinking and answer from the response
    # Implement this function based on your specific needs
    return "", response

# Initialize components
check_openai_api_key()
vector_store = initialize_pinecone()
llm, embeddings = initialize_llm()
prompt = initialize_prompt()

# Streamlit UI for uploading PDF and asking questions
pdfs_directory = 'data/'
uploaded_file = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=False)

if uploaded_file:
    upload_pdf(uploaded_file)
    documents = load_pdf(pdfs_directory + uploaded_file.name)
    chunked_documents = split_text(documents)
    index_docs(chunked_documents)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Prompt for user input and save to chat history
if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                related_documents = retrieve_docs(prompt)
                response = answer_question(prompt, related_documents)
                st.write(response)
                message = {"role": "assistant", "content": response}
                st.session_state.messages.append(message)
else:
    # Welcome message when no file is uploaded
    st.markdown("""
        <div style='text-align: center; padding: 3rem; color: #6B7280; background-color: #F3F4F6; border-radius: 1rem;'>
            <h2 style='color: #1E3A8A; margin-bottom: 1rem;'>Welcome to DeepSeek R1 RAG</h2>
            <p style='font-size: 1.1rem; margin-bottom: 1rem;'>👈 Start by uploading a PDF document in the sidebar</p>
            <p style='font-size: 0.9rem;'>Supported format: PDF</p>
        </div>
    """, unsafe_allow_html=True)
