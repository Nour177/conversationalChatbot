"""
FAISS-based retriever for document search.
Handles loading the FAISS index and retrieving relevant documents based on queries.
"""

import os
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from process_data import load_faiss_index, EMBEDDING_MODEL_NAME

# Configuration
FAISS_INDEX_DIR = Path("data/processed/faiss_index")
k_default = os.getenv("RETRIEVER_K", 5)


class FAISSRetriever:
    """FAISS-based retriever for semantic document search."""
    
    def __init__(self, index_dir: Path = FAISS_INDEX_DIR):
        """
        Initialize the FAISS retriever.
        
        Args:
            index_dir: Directory containing the FAISS index and metadata
        """
        self.index_dir = index_dir
        self.index = None
        self.chunks = None
        self.embedding_model = None
        
        self._load_index()
        self._load_embedding_model()
    
    def _load_index(self):
        """Load the FAISS index and metadata."""
        print("Loading FAISS index and metadata...")
        index, metadata = load_faiss_index(self.index_dir)
        
        if index is None or metadata is None:
            raise ValueError(
                "FAISS index not found. Please run process_data.py first."
            )
        
        self.index = index
        self.chunks = metadata["chunks"]
        print(f"✓ Index loaded with {self.index.ntotal} vectors\n")
    
    def _load_embedding_model(self):
        """Load the embedding model for query encoding."""
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("✓ Embedding model loaded\n")
    
    def retrieve_documents(self, query: str, k: int = k_default) -> str:
        """
        Retrieve relevant documents from FAISS index based on query.
        
        Args:
            query: User's question
            k: Number of documents to retrieve
            
        Returns:
            Formatted string of relevant document chunks
        """
        # Encode the query
        query_embedding = self.embedding_model.encode(
            [query], 
            convert_to_numpy=True, 
            show_progress_bar=False
        )
        
        # Normalize the query embedding (same as in process_data.py)
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index
        k = min(k, self.index.ntotal)  # Ensure k doesn't exceed total vectors
        distances, indices = self.index.search(
            query_embedding.astype('float32'), 
            k
        )
        
        # Retrieve relevant chunks
        relevant_chunks = []
        for idx in indices[0]:
            if idx < len(self.chunks):
                relevant_chunks.append(self.chunks[idx].page_content)
        
        # Format the retrieved documents
        formatted_docs = "\n\n".join([
            f"Document {i+1}:\n{chunk}" 
            for i, chunk in enumerate(relevant_chunks)
        ])
        return formatted_docs


# Convenience function for backward compatibility
def get_retriever(index_dir: Path = FAISS_INDEX_DIR) -> FAISSRetriever:
    """
    Get a FAISS retriever instance.
    
    Args:
        index_dir: Directory containing the FAISS index and metadata
        
    Returns:
        FAISSRetriever instance
    """
    return FAISSRetriever(index_dir)
