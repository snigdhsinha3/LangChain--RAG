# document_processor.py
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


# --- Configuration ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
#"BAAI/bge-large-en-v1.5"

MANUALS_DIR = "docs" # Directory for the custom data
FAISS_PERSIST_DIR = "./FAISS_database_rag" # Directory for FAISS persistence

# --- Load Documents and Split ---
def load_and_split_documents(manuals_dir = MANUALS_DIR):
    if not os.path.exists(manuals_dir) or not os.listdir(manuals_dir):
        print(f"Warning: '{manuals_dir}' directory not found or is empty. RAG will not have documents")
        return []  # Returns empty list if no docs 
    
    # Load documents from manual_dir
    documents = []
    print(f"Loading documents from '{manuals_dir}'...")
    for file_name in os.listdir(manuals_dir):
        file_path = os.path.join(manuals_dir, file_name)
        print(f"Processing file: {file_name}")
        try:
            if file_name.endswith(".txt"):
                loader = TextLoader(file_path, encoding ='utf-8')
            elif file_name.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            else:
                print(f"Skipping the unsupported file: {file_name}")
                continue

            docs = loader.load() # This returns the List [Document]
            documents.extend(docs)
            print(f"Loaded: {file_name}")

        except Exception as e:
            print(f"Error loading file: {file_name}: {e}")
        
    # Split the documents into chunk
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Split documetns into {len(chunks)} chunks.")
    return chunks

# --- Create Embeddings (Global instance) ---
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
print(f"Embedding model '{EMBEDDING_MODEL_NAME}' loaded in document_processor.py.")
    