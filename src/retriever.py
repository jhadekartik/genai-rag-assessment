"""RAG Pipeline orchestration - Main retrieval engine"""

import logging
from typing import List, Dict, Tuple
import numpy as np
from datetime import datetime

from .embeddings import EmbeddingModel
from .vector_store import FAISSVectorStore
from .query_expander import MockQueryExpander
from .config import (
    EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION,
    DEFAULT_TOP_K, SIMILARITY_METRIC
)

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete Retrieval Augmented Generation (RAG) pipeline.
    
    Supports two retrieval strategies:
    - Strategy A: Raw Vector Search (direct embedding similarity)
    - Strategy B: AI-Enhanced Retrieval (query expansion + embedding similarity)
    """
    
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        vector_dim: int = EMBEDDING_DIMENSION,
        use_query_expansion: bool = True
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            model_name: Sentence-transformer model name
            vector_dim: Embedding dimension
            use_query_expansion: Enable Strategy B (query expansion)
        """
        self.model_name = model_name
        self.vector_dim = vector_dim
        self.use_query_expansion = use_query_expansion
        
        # Initialize components
        self.embedding_model = EmbeddingModel(model_name)
        self.vector_store = FAISSVectorStore(vector_dim)
        self.query_expander = MockQueryExpander()
        
        self.documents = []
        self.ingested = False
        
        logger.info(
            f"Initialized RAGPipeline (model={model_name}, "
            f"expansion={'enabled' if use_query_expansion else 'disabled'})"
        )
    
    def ingest_documents(self, documents: List[str], chunk_size: int = None) -> None:
        """
        Ingest raw documents into the RAG pipeline.
        
        Args:
            documents: List of raw document texts
            chunk_size: Optional chunk size for long documents
        """
        if not documents:
            raise ValueError("No documents provided for ingestion")
        
        logger.info(f"Starting document ingestion ({len(documents)} documents)...")
        
        # Optional: Chunk long documents
        if chunk_size:
            documents = self._chunk_documents(documents, chunk_size)
        
        # Generate embeddings
        embeddings = self.embedding_model.embed_documents(documents)
        
        # Add to vector store
        self.vector_store.add_documents(documents, embeddings)
        
        self.documents = documents
        self.ingested = True
        
        logger.info(f"Document ingestion complete. Total documents: {len(documents)}")
    
    def retrieve_strategy_a(self, query: str, top_k: int = DEFAULT_TOP_K) -> Dict:
        """
        Strategy A: Raw Vector Search
        Direct embedding-based similarity search.
        
        Args:
            query: User query string
            top_k: Number of top results to return
            
        Returns:
            Dictionary with strategy A results
        """
        if not self.ingested:
            raise ValueError("Pipeline not initialized. Call ingest_documents() first.")
        
        logger.info(f"Strategy A: Retrieving for query: '{query}'")
        
        # Embed query
        query_embedding = self.embedding_model.embed_text(query)
        
        # Search
        results = self.vector_store.search(query_embedding[0], top_k=top_k)
        
        return {
            "strategy": "A_RAW_VECTOR_SEARCH",
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "results": [
                {
                    "rank": i + 1,
                    "document_id": doc_id,
                    "similarity_score": float(score),
                    "document_excerpt": doc_text[:200] + "..." if len(doc_text) > 200 else doc_text
                }
                for i, (doc_id, score, doc_text) in enumerate(results)
            ],
            "full_documents": [doc_text for _, _, doc_text in results]
        }
    
    def retrieve_strategy_b(self, query: str, top_k: int = DEFAULT_TOP_K) -> Dict:
        """
        Strategy B: AI-Enhanced Retrieval
        Uses query expansion to rewrite the query before embedding.
        
        Args:
            query: User query string
            top_k: Number of top results to return
            
        Returns:
            Dictionary with strategy B results
        """
        if not self.ingested:
            raise ValueError("Pipeline not initialized. Call ingest_documents() first.")
        
        logger.info(f"Strategy B: Retrieving for query: '{query}'")
        
        # Expand query
        expanded_query = self.query_expander.expand_query(query)
        
        # Embed expanded query
        expanded_embedding = self.embedding_model.embed_text(expanded_query)
        
        # Search
        results = self.vector_store.search(expanded_embedding[0], top_k=top_k)
        
        # Get expansion explanation
        expansion_info = self.query_expander.explain_expansion(query, expanded_query)
        
        return {
            "strategy": "B_AI_ENHANCED_RETRIEVAL",
            "original_query": query,
            "expanded_query": expanded_query,
            "expansion_info": expansion_info,
            "timestamp": datetime.now().isoformat(),
            "results": [
                {
                    "rank": i + 1,
                    "document_id": doc_id,
                    "similarity_score": float(score),
                    "document_excerpt": doc_text[:200] + "..." if len(doc_text) > 200 else doc_text
                }
                for i, (doc_id, score, doc_text) in enumerate(results)
            ],
            "full_documents": [doc_text for _, _, doc_text in results]
        }
    
    def compare_strategies(self, query: str, top_k: int = DEFAULT_TOP_K) -> Dict:
        """
        Compare both retrieval strategies for a single query.
        
        Args:
            query: User query string
            top_k: Number of results per strategy
            
        Returns:
            Dictionary with comparison between strategies
        """
        logger.info(f"Comparing strategies for query: '{query}'")
        
        # Run both strategies
        results_a = self.retrieve_strategy_a(query, top_k)
        results_b = self.retrieve_strategy_b(query, top_k)
        
        # Calculate metrics
        scores_a = [r["similarity_score"] for r in results_a["results"]]
        scores_b = [r["similarity_score"] for r in results_b["results"]]
        
        avg_score_a = np.mean(scores_a) if scores_a else 0
        avg_score_b = np.mean(scores_b) if scores_b else 0
        
        # Check result overlap
        docs_a = set(r["document_id"] for r in results_a["results"])
        docs_b = set(r["document_id"] for r in results_b["results"])
        overlap = len(docs_a & docs_b) / len(docs_a | docs_b) if (docs_a | docs_b) else 0
        
        comparison = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "strategy_a": results_a,
            "strategy_b": results_b,
            "metrics": {
                "strategy_a_avg_similarity": float(avg_score_a),
                "strategy_b_avg_similarity": float(avg_score_b),
                "similarity_improvement": float(avg_score_b - avg_score_a),
                "improvement_percentage": float((avg_score_b - avg_score_a) / (avg_score_a + 1e-8) * 100),
                "result_overlap_ratio": float(overlap),
                "docs_unique_to_a": list(docs_a - docs_b),
                "docs_unique_to_b": list(docs_b - docs_a)
            }
        }
        
        return comparison
    
    def batch_compare(self, queries: List[str], top_k: int = DEFAULT_TOP_K) -> List[Dict]:
        """
        Compare strategies for multiple queries.
        
        Args:
            queries: List of query strings
            top_k: Number of results per strategy
            
        Returns:
            List of comparison results
        """
        comparisons = [self.compare_strategies(q, top_k) for q in queries]
        logger.info(f"Completed batch comparison for {len(queries)} queries")
        return comparisons
    
    def get_pipeline_stats(self) -> Dict:
        """Get statistics about the pipeline"""
        return {
            "embedding_model": self.embedding_model.get_model_info(),
            "vector_store": self.vector_store.get_stats(),
            "documents_ingested": self.ingested,
            "query_expansion_enabled": self.use_query_expansion
        }
    
    @staticmethod
    def _chunk_documents(documents: List[str], chunk_size: int) -> List[str]:
        """
        Split long documents into chunks.
        
        Args:
            documents: List of documents
            chunk_size: Approximate chunk size in characters
            
        Returns:
            List of document chunks
        """
        chunks = []
        for doc in documents:
            if len(doc) <= chunk_size:
                chunks.append(doc)
            else:
                # Split on sentence boundaries if possible
                sentences = doc.split(". ")
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= chunk_size:
                        current_chunk += sentence + ". "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + ". "
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
        
        return chunks
