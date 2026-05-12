"""FAISS-based vector database for efficient semantic search"""

import numpy as np
import faiss
from typing import List, Tuple, Dict
import logging
import json
import pickle
import os

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    Lightweight local vector database using Facebook AI Similarity Search (FAISS).
    Optimized for semantic similarity search with cosine distance metric.
    """
    
    def __init__(self, embedding_dim: int, use_gpu: bool = False):
        """
        Initialize FAISS vector store.
        
        Args:
            embedding_dim: Dimension of embeddings
            use_gpu: Whether to use GPU acceleration (if available)
        """
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu
        self.documents = []  # Store original texts
        self.embeddings = None  # numpy array of embeddings
        
        # Create FAISS index for cosine similarity
        # Using Flat index for exact search (suitable for local datasets)
        quantizer = faiss.IndexFlatL2(embedding_dim)
        self.index = faiss.IndexIDMap(quantizer)
        
        self.document_count = 0
        logger.info(f"Initialized FAISS vector store (dim={embedding_dim}, GPU={use_gpu})")
    
    def add_documents(self, documents: List[str], embeddings: np.ndarray) -> None:
        """
        Add documents and their embeddings to the vector store.
        
        Args:
            documents: List of document texts
            embeddings: numpy array of shape (len(documents), embedding_dim)
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )
        
        # Normalize embeddings for cosine similarity
        normalized_embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Convert to float32 for FAISS
        normalized_embeddings = normalized_embeddings.astype(np.float32)
        
        # Generate IDs
        start_id = self.document_count
        ids = np.arange(start_id, start_id + len(documents), dtype=np.int64)
        
        # Add to index
        self.index.add_with_ids(normalized_embeddings, ids)
        
        # Store documents and embeddings
        self.documents.extend(documents)
        if self.embeddings is None:
            self.embeddings = normalized_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, normalized_embeddings])
        
        self.document_count += len(documents)
        logger.info(f"Added {len(documents)} documents. Total: {self.document_count}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Tuple[int, float, str]]:
        """
        Search for similar documents using cosine similarity.
        
        Args:
            query_embedding: Query embedding (1D array)
            top_k: Number of top results to return
            
        Returns:
            List of tuples (document_id, similarity_score, document_text)
        """
        if self.document_count == 0:
            logger.warning("Vector store is empty")
            return []
        
        # Normalize query embedding for cosine similarity
        query_normalized = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        query_normalized = query_normalized.astype(np.float32).reshape(1, -1)
        
        # Search (FAISS returns distances, we convert to similarity)
        distances, indices = self.index.search(query_normalized, min(top_k, self.document_count))
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx == -1:  # Invalid index
                continue
            
            # Convert L2 distance to cosine similarity
            # For normalized vectors: L2_distance = 2 * (1 - cosine_similarity)
            similarity = 1 - (distance / 2)
            
            results.append((
                int(idx),
                float(similarity),
                self.documents[int(idx)]
            ))
        
        return results
    
    def batch_search(self, query_embeddings: np.ndarray, top_k: int = 3) -> List[List[Tuple[int, float, str]]]:
        """
        Batch search for multiple queries.
        
        Args:
            query_embeddings: Query embeddings (shape: n_queries x embedding_dim)
            top_k: Number of top results per query
            
        Returns:
            List of result lists (one per query)
        """
        # Normalize query embeddings
        query_normalized = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
        query_normalized = query_normalized.astype(np.float32)
        
        distances, indices = self.index.search(query_normalized, min(top_k, self.document_count))
        
        batch_results = []
        for query_idx in range(len(indices)):
            results = []
            for idx, distance in zip(indices[query_idx], distances[query_idx]):
                if idx == -1:
                    continue
                
                similarity = 1 - (distance / 2)
                results.append((
                    int(idx),
                    float(similarity),
                    self.documents[int(idx)]
                ))
            
            batch_results.append(results)
        
        return batch_results
    
    def get_document(self, doc_id: int) -> str:
        """Get a document by ID"""
        if 0 <= doc_id < len(self.documents):
            return self.documents[doc_id]
        raise IndexError(f"Document ID {doc_id} not found")
    
    def get_stats(self) -> Dict:
        """Get vector store statistics"""
        return {
            "document_count": self.document_count,
            "embedding_dimension": self.embedding_dim,
            "index_type": "FAISS-IndexIDMap-Flat-L2",
            "similarity_metric": "cosine",
            "memory_usage_mb": (
                self.embeddings.nbytes / (1024 * 1024) if self.embeddings is not None else 0
            )
        }
    
    def save(self, filepath: str) -> None:
        """Save vector store to disk"""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        
        data = {
            "documents": self.documents,
            "embeddings": self.embeddings,
            "embedding_dim": self.embedding_dim,
            "document_count": self.document_count
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        
        logger.info(f"Vector store saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load vector store from disk"""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        
        self.documents = data["documents"]
        self.embeddings = data["embeddings"]
        self.embedding_dim = data["embedding_dim"]
        self.document_count = data["document_count"]
        
        # Rebuild FAISS index
        quantizer = faiss.IndexFlatL2(self.embedding_dim)
        self.index = faiss.IndexIDMap(quantizer)
        
        ids = np.arange(self.document_count, dtype=np.int64)
        self.index.add_with_ids(self.embeddings.astype(np.float32), ids)
        
        logger.info(f"Vector store loaded from {filepath}")
