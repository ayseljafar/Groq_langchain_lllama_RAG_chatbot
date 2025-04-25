import os
from typing import List
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

class DocumentProcessor:
    def __init__(self):
        self.persist_directory = os.getenv('PERSIST_DIRECTORY', 'db')
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
    def load_documents(self, directory: str = "documents") -> List:
        """Load documents from the specified directory"""
        loader = DirectoryLoader(
            directory,
            glob="**/*.*",
            loader_cls=UnstructuredFileLoader
        )
        documents = loader.load()
        return documents
    
    def split_documents(self, documents: List, chunk_size: int = 1000, chunk_overlap: int = 200) -> List:
        """Split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        splits = text_splitter.split_documents(documents)
        return splits
    
    def create_vector_store(self, documents: List) -> Chroma:
        """Create or load vector store"""
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        vector_store.persist()
        return vector_store
    
    def process_documents(self) -> Chroma:
        """Process documents and create vector store"""
        documents = self.load_documents()
        splits = self.split_documents(documents)
        vector_store = self.create_vector_store(splits)
        return vector_store

if __name__ == "__main__":
    processor = DocumentProcessor()
    vector_store = processor.process_documents()
    print(f"Processed documents and created vector store at {processor.persist_directory}") 