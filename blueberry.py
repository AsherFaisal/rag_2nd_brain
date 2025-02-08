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
from langgraph.graph import START, StateGraph
from IPython.display import Image, display
from langchain_core.documents import Document
from typing_extensions import List, TypedDict

load_dotenv()

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

def index_docs(documents):
    vector_store.add_documents(documents)

def retrieve_docs(query):
    return vector_store.similarity_search(query)

def initialize_prompt():
    return hub.pull("rlm/rag-prompt")

def example_messages(prompt):
    messages = prompt.invoke(
        {"context": "(context goes here)", "question": "(question goes here)"}
    ).to_messages()
    assert len(messages) == 1
    return messages[0].content

def initialize_llm():
    llm = init_chat_model("gpt-4o-mini", model_provider="openai")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return llm, embeddings

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
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})

# Initialize components
check_openai_api_key()
vector_store = initialize_pinecone()
llm, embeddings = initialize_llm()
prompt = initialize_prompt()

# Build and compile the state graph
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Display the graph
display(Image(graph.get_graph().draw_mermaid_png()))

# Invoke the graph with a sample question
result = graph.invoke({"question": "What is Task Decomposition?"})

# Print the result
print(f'Context: {result["context"]}\n\n')
print(f'Answer: {result["answer"]}')