# vectorstore_manager.py
"""
vectorstore_manager.py

Manages the creation, loading, saving, and updating of a FAISS-based vector store
for Retrieval-Augmented Generation (RAG) workflows using LangChain.

This module interacts with preprocessed document chunks (from `document_processor`)
and generates a persistent FAISS index for fast, similarity-based retrieval.

Features:
    - Automatic loading or fallback to building FAISS vector store
    - Persistent saving of index to local filesystem
    - Manual update capability via `update_vectorstore()`
    - Returns retrievers for integration into LangChain RAG chains

Attributes:
    FAISS_INDEX_DIR (str): Directory to persist FAISS index files.
    FAISS_INDEX_NAME (str): Filename for saved index.
    _retriever_instace (Retriever): Singleton retriever object for global reuse.

Functions:
    _build_and_save_vectorstore(manuals_dir: str) -> Optional[Retriever]:
        Loads, splits documents, builds FAISS index, saves locally, and returns retriever.
    
    get_retriever() -> Optional[Retriever]:
        Returns a singleton retriever. Attempts to load persisted index; 
        falls back to building if index unavailable or corrupted.

    update_vectorstore() -> str:
        Rebuilds FAISS vector store from source documents. Returns status string.

Notes:
    - Requires `document_processor.load_and_split_documents` to return valid chunk list.
    - Embeddings object must match those used during saving/loading.
    - Dangerous deserialization is enabled due to FAISS limitations—review for security in prod.

Example:
    >>> from vectorstore_manager import get_retriever
    >>> retriever = get_retriever()
    >>> retriever.invoke("How do I reset the device?")
"""

import os
import logging
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List, Optional

# Import components from document_procecssor
from document_pocessor import load_and_split_documents, embeddings, MANUALS_DIR

# --- Configuration for FAISS Persistence ---
FAISS_INDEX_DIR = "./faiss_index" # Directory to save the FAISS index
FAISS_INDEX_NAME = "manual_faiss_index" # Name of the index file

# --- Logger for this module ---
logger = logging.getLogger(__name__)

# Global variable to hold the retriever instance
_retriever_instance = None

def _build_and_save_vectorstore(manuals_dir = MANUALS_DIR):
    """
    Loads, splits documents, builds a new FAISS vector store, and saves it. 
    """
    logger.info(f"Building new FAISS vector store from '{manuals_dir}")
    chunks = load_and_split_documents(manuals_dir)

    if not chunks:
        logger.warning("No document splits available. Cannot build a vector store.")
        return None
    
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Create the directory if it doesn't exist
    if not os.path.exists(FAISS_INDEX_DIR):
        os.makedirs(FAISS_INDEX_DIR)
        logger.info(f"Created FAISS index directory: '{FAISS_INDEX_DIR}'")

        # Save the vector store locally
        save_path = os.path.join(FAISS_INDEX_DIR, FAISS_INDEX_NAME)
        vectorstore.save_local(save_path)
        logger.info(f"FAISS vector store saved to: '{save_path}'")

        return vectorstore.as_retriever(search_kwargs={"k": 3})
    

def get_retriever():
    """
    Attempts to load an existing FAISS vector store. If not found or fails,
    it builds a new one and saves it.
    Returns the retriever instance.
    """

    global _retriever_instance

    if _retriever_instance:
        logger.info("Retriever instance already exists. Returning existing one.")
        return _retriever_instance
    
    load_path = os.path.join(FAISS_INDEX_DIR, FAISS_INDEX_NAME)

    if os.path.exists(load_path):
        try:
            logger.info(f"Attempting to load FAISS vector store from '{load_path}'...")
            #Ensure embeddings are passed when loading
            vectorstore = FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)
            _retriever_instance = vectorstore.as_retriever(search_kwargs={"k":3})
            logger.info("FAISS vector loaded succesfully.")
            return _retriever_instance
        except Exception as e:
            logger.error(f"Failed to load FAISS vector store from {load_path}: {e}")
            logger.warning("Falling back to building a new vector store.")
            _retriever_instance = _build_and_save_vectorstore(MANUALS_DIR)
            return _retriever_instance
    
    else:
        logger.info(f"No existing FAISS vector store found at '{load_path}'. Building a new one...")
        _retriever_instance = _build_and_save_vectorstore(MANUALS_DIR)
        return _retriever_instance
    

# --- Funciton to manually update the vector store ---
def update_vectorstore():
    """
    Manually triggers the process to rebuild the vector store from documenbts
    and save it. Updates the global retirever instance.
    """
    global _retriever_instance
    logger.info("Manual update of vector store initiated.")
    _retriever_instance = _build_and_save_vectorstore(MANUALS_DIR)
    if _retriever_instance:
        logger.info("Vector store updated successully.")
        return "Vector store updated successfully!"
    else:
        logger.error("Failed to update vector store. Check logs fo details.")
        return "Failed to updated vector store. Check logs."
    
# --- Initialize retriever on module load (default behavior) ---
# This ensures that the retiever is ready when imported by other modules
_retriever_instance = get_retriever()