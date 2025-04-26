# RAG-Powered Q&A Chatbot

A question-answering chatbot that uses Retrieval-Augmented Generation (RAG) to provide accurate answers based on your documents. Built with Streamlit, LangChain, and Groq.

## Features

- 🤖 Powered by Groq's LLama 3.3 70B model
- 📚 Document processing with automatic chunking and embedding
- 🔍 Semantic search using ChromaDB vector store
- 💬 Interactive chat interface with Streamlit
- 🔄 Automatic source citations
- ⚡ Fast response times with Groq's API
- 🛡️ Error handling with automatic retries

## Prerequisites

- Python 3.11 or higher
- A Groq API key (get one at [console.groq.com](https://console.groq.com))
- (Optional) Homebrew for macOS users

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ayseljafar/Groq_langchain_lllama_RAG_chatbot
cd langchain-rag
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install additional system dependencies (macOS):
```bash
brew install libmagic  # Required for document processing
```

5. Set up environment variables:
   - Copy the template: `cp template.env .env`
   - Add your Groq API key to the `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
PERSIST_DIRECTORY=db
```

## Usage

1. Add your documents:
   - Place your documents (PDF, TXT, etc.) in the `documents` folder
   - Supported formats include PDF, TXT, DOCX, and more

2. Process the documents:
```bash
python document_processor.py
```

3. Run the chatbot:
```bash
streamlit run app.py
```

4. Open your browser and navigate to `http://localhost:8501`

## Project Structure

```
langchain-rag/
├── app.py              # Main Streamlit application
├── document_processor.py # Document processing and embedding
├── documents/          # Directory for your documents
├── db/                # Vector store database
├── requirements.txt    # Python dependencies
├── .env               # Environment variables
└── README.md          # This file
```

## How It Works

1. **Document Processing**:
   - Documents are loaded from the `documents` folder
   - Text is split into chunks with overlap for context preservation
   - Chunks are embedded using HuggingFace's sentence transformers
   - Embeddings are stored in a ChromaDB vector store

2. **Question Answering**:
   - User questions are processed through the RAG pipeline
   - Relevant document chunks are retrieved using semantic search
   - Groq's LLM generates answers based on the retrieved context
   - Sources are automatically cited for transparency

## Troubleshooting

- If you encounter document processing issues, ensure `libmagic` is installed
- For API errors, check your Groq API key and internet connection
- The system includes automatic retry mechanisms for API calls

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
