# RAG-Powered Q&A Chatbot

A document-aware chatbot powered by Google's Gemma 2B model and LangChain's RAG (Retrieval Augmented Generation) capabilities.

## Features

- Document-aware responses using RAG
- Powered by Google's Gemma 2B model
- Source citations for answers
- Support for multiple document formats
- Persistent chat history
- Modern Streamlit UI

## Setup

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `template.env` to `.env` and add your Hugging Face token:
   ```bash
   cp template.env .env
   ```
   Then edit `.env` and add your Hugging Face token.

## Usage

1. Place your documents in the `documents` folder. Supported formats include:
   - PDF
   - TXT
   - DOCX
   - And more (see Unstructured documentation)

2. Process the documents:
   ```bash
   python document_processor.py
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

4. Open your browser and navigate to the provided URL (usually http://localhost:8501)

## How it Works

1. Documents are processed and split into chunks
2. Text chunks are embedded using Sentence Transformers
3. Embeddings are stored in a Chroma vector store
4. When you ask a question:
   - The question is embedded and similar chunks are retrieved
   - Retrieved context is sent to Gemma 2B along with your question
   - The model generates a response based on the context
   - Sources are cited for transparency

## Controls

- **Reprocess Documents**: Click this button in the sidebar to reprocess documents if you've added new ones
- **Clear Chat**: Clears the chat history and starts a new conversation

## Notes

- The quality of responses depends on the documents provided
- The model works best with well-structured, clear documents
- Processing time depends on document size and quantity 