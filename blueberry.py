import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain import hub
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from typing_extensions import List, TypedDict

load_dotenv()

template = """
You are a helpful AI assistant. Your task is to:
1. Always cite your sources by referring to the specific parts of the document you used.
2. If you don't know the answer, just say that you don't know. 
3. Keep responses concise and focused.
Question: {question} 
Context: {context} 
Answer:
"""

pdfs_directory = 'data/'

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

def check_openai_api_key():
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")

def initialize_pinecone():
    pc_token = os.environ.get("PINECONE_API_KEY")
    pc = Pinecone(api_key=pc_token)
    index = pc.Index("knowledgeset")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return PineconeVectorStore(embedding=embeddings, index=index)

def initialize_prompt():
    return hub.pull("rlm/rag-prompt")

def initialize_llm():
    llm = init_chat_model("o3-mini", model_provider="openai")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return llm, embeddings

def index_docs(documents):
    vector_store.add_documents(documents)

def retrieve_docs(query):
    return vector_store.similarity_search(query)

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt_template = ChatPromptTemplate.from_template(template)
    messages = prompt_template.invoke({"question": question, "context": context})
    response = llm.invoke(messages)
    return response.content

# Initialize components
check_openai_api_key()
vector_store = initialize_pinecone()
llm, embeddings = initialize_llm()
prompt = initialize_prompt()


# Streamlit UI for uploading PDF and asking questions
uploaded_file = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=False)

if uploaded_file:
    upload_pdf(uploaded_file)
    documents = load_pdf(pdfs_directory + uploaded_file.name)
    chunked_documents = split_text(documents)
    index_docs(chunked_documents)

question = st.chat_input()

if question:
    st.chat_message("user").write(question)
    related_documents = retrieve_docs(question)
    answer = answer_question(question, related_documents)
    st.chat_message("assistant").write(answer)

