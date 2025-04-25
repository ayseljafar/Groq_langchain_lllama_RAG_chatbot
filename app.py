import streamlit as st
import os
from dotenv import load_dotenv
from document_processor import DocumentProcessor
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
import time
from typing import Dict, Any, Callable
from functools import wraps
import chromadb
from chromadb.config import Settings

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="RAG-Powered Q&A Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Check for Groq API key
if 'GROQ_API_KEY' not in os.environ:
    st.error("Please set your Groq API key in .env file")
    st.stop()

def retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 1,
    exponential_base: float = 2,
    error_types: tuple = (Exception,),
) -> Callable:
    """Retry decorator with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except error_types as e:
                    if i == max_retries - 1:  # Last iteration
                        raise e
                    
                    st.warning(f"Attempt {i + 1} failed. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= exponential_base
            
            return None  # If all retries fail
        return wrapper
    return decorator

# Create header
st.title("🤖 RAG-Powered Q&A Chatbot")
st.markdown("Ask me anything about your documents! I'm powered by Groq's Mixtral model with RAG.")

@st.cache_resource
def initialize_rag():
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Initialize ChromaDB client
    persist_directory = os.getenv('PERSIST_DIRECTORY', 'db')
    chroma_client = chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
    
    # Load vector store
    vector_store = Chroma(
        client=chroma_client,
        collection_name="document_store",
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    
    # Initialize Groq LLM with retry wrapper
    llm = ChatGroq(
        temperature=0.7,
        model_name="llama-3.3-70b-versatile",
        groq_api_key=os.environ['GROQ_API_KEY'],
        max_tokens=2048
    )
    
    # Create chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        verbose=True
    )
    
    return chain

# Initialize RAG system
chain = initialize_rag()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
if prompt := st.chat_input("What's your question about the documents?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            with st.spinner("Thinking..."):
                # Define the chain response function with retry
                @retry_with_exponential_backoff(max_retries=3)
                def get_chain_response(query: str, history: list) -> Dict[str, Any]:
                    return chain({
                        "question": query,
                        "chat_history": history
                    })
                
                response = get_chain_response(prompt, st.session_state.chat_history)
                
                answer = response["answer"]
                source_docs = response["source_documents"]
                
                # Format response with sources
                full_response = f"{answer}\n\n**Sources:**\n"
                for i, doc in enumerate(source_docs, 1):
                    full_response += f"{i}. {doc.metadata.get('source', 'Unknown source')}\n"
                
                # Display and save assistant response
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
                # Update chat history
                st.session_state.chat_history.append((prompt, answer))
                
        except Exception as e:
            error_message = f"An error occurred: {str(e)}\n\nPlease try again or contact support if the problem persists."
            message_placeholder.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

# Add a sidebar with information and controls
with st.sidebar:
    st.title("About")
    st.markdown("""
    This is a RAG-powered Q&A chatbot that can answer questions about your documents.
    
    Features:
    - Document-aware responses using RAG
    - Powered by Groq's Mixtral-8x7B model
    - Source citations
    - Persistent chat history
    - Markdown support
    - Automatic retry on API errors
    
    Note: Responses are generated using AI and may not always be accurate.
    """)
    
    # Add document processing button
    if st.button("Reprocess Documents"):
        with st.spinner("Processing documents..."):
            processor = DocumentProcessor()
            processor.process_documents()
            st.success("Documents processed successfully!")
    
    # Add a clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun() 