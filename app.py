import streamlit as st
import torch
from transformers import pipeline
import os
from huggingface_hub import login
from dotenv import load_dotenv
from document_processor import DocumentProcessor
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFacePipeline

# Load environment variables
load_dotenv()

# Check for Hugging Face token
if 'HF_TOKEN' not in os.environ:
    st.error("Please set your Hugging Face token in .env file")
    st.stop()

# Login to Hugging Face
login(token=os.environ['HF_TOKEN'])

# Set page config
st.set_page_config(
    page_title="RAG-Powered Q&A Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Create header
st.title("🤖 RAG-Powered Q&A Chatbot")
st.markdown("Ask me anything about your documents! I'm powered by Google's Gemma 2B model with RAG.")

@st.cache_resource
def initialize_rag():
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Load vector store
    vector_store = Chroma(
        persist_directory=os.getenv('PERSIST_DIRECTORY', 'db'),
        embedding_function=embeddings
    )
    
    # Initialize LLM
    llm = pipeline(
        "text-generation",
        model="google/gemma-2b-it",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    hf_llm = HuggingFacePipeline(
        pipeline=llm,
        model_kwargs={
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.95,
            "repetition_penalty": 1.1
        }
    )
    
    # Create chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=hf_llm,
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
        
        with st.spinner("Thinking..."):
            # Get response from RAG chain
            response = chain({
                "question": prompt,
                "chat_history": st.session_state.chat_history
            })
            
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

# Add a sidebar with information and controls
with st.sidebar:
    st.title("About")
    st.markdown("""
    This is a RAG-powered Q&A chatbot that can answer questions about your documents.
    
    Features:
    - Document-aware responses
    - Source citations
    - Persistent chat history
    - Markdown support
    
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