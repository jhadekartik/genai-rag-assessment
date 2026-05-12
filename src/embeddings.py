"""Embedding model wrapper for semantic text representation"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union
import logging

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Wrapper around sentence-transformers to simulate Vertex AI's 
    text-embedding-gecko behavior for local development.
    
    Similarity Metrics:
    - Cosine: Angle-based similarity (magnitude-independent)
    - Euclidean: Distance in vector space
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Initialized {model_name} with dimension {self.embedding_dim}")
    
    def embed_text(self, text: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            text: Single string or list of strings
            batch_size: Batch size for processing
            
        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        if isinstance(text, str):
            text = [text]
        
        embeddings = self.model.encode(
            text,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        return embeddings
    
    def embed_documents(self, documents: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed a list of documents efficiently.
        
        Args:
            documents: List of document texts
            batch_size: Batch size for processing
            
        Returns:
            numpy array of embeddings
        """
        logger.info(f"Embedding {len(documents)} documents...")
        embeddings = self.embed_text(documents, batch_size=batch_size)
        logger.info(f"Successfully embedded {len(documents)} documents")
        return embeddings
    
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        Range: [-1, 1] where 1 is identical, 0 is orthogonal, -1 is opposite.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        # Normalize vectors
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        return float(np.dot(vec1_norm, vec2_norm))
    
    @staticmethod
    def cosine_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine distance (1 - cosine_similarity).
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine distance (0 to 2)
        """
        similarity = EmbeddingModel.cosine_similarity(vec1, vec2)
        return 1 - similarity
    
    @staticmethod
    def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute Euclidean distance between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Euclidean distance
        """
        return float(np.linalg.norm(vec1 - vec2))
    
    def get_model_info(self) -> dict:
        """Return model information"""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "model_type": "sentence-transformers",
            "notes": "Local equivalent to Vertex AI text-embedding-gecko"
        }
