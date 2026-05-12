"""Integration tests for RAG pipeline"""

import pytest
import numpy as np
from src.retriever import RAGPipeline
from src.embeddings import EmbeddingModel


@pytest.fixture
def sample_documents():
    """Fixture providing sample documents"""
    return [
        "The system handles peak loads through auto-scaling and load balancing strategies",
        "Security is implemented with encryption, authentication, and access control",
        "Architecture is based on microservices with distributed processing capabilities",
        "Database optimization uses indexing and query caching for performance",
        "API rate limiting prevents abuse and ensures fair resource usage"
    ]


@pytest.fixture
def rag_pipeline(sample_documents):
    """Fixture providing initialized RAG pipeline"""
    rag = RAGPipeline(use_query_expansion=True)
    rag.ingest_documents(sample_documents)
    return rag


class TestRAGPipeline:
    """Integration tests for RAGPipeline"""
    
    def test_initialization(self):
        """Test RAG pipeline initialization"""
        rag = RAGPipeline()
        
        assert rag.embedding_model is not None
        assert rag.vector_store is not None
        assert rag.query_expander is not None
        assert rag.ingested is False
    
    def test_document_ingestion(self, rag_pipeline, sample_documents):
        """Test document ingestion"""
        assert rag_pipeline.ingested is True
        assert rag_pipeline.vector_store.document_count == len(sample_documents)
        assert len(rag_pipeline.documents) == len(sample_documents)
    
    def test_empty_ingestion_raises_error(self):
        """Test that empty document list raises error"""
        rag = RAGPipeline()
        
        with pytest.raises(ValueError):
            rag.ingest_documents([])
    
    def test_strategy_a_retrieval(self, rag_pipeline):
        """Test Strategy A (raw vector search)"""
        query = "How does the system handle high traffic?"
        
        results = rag_pipeline.retrieve_strategy_a(query, top_k=3)
        
        assert results["strategy"] == "A_RAW_VECTOR_SEARCH"
        assert results["query"] == query
        assert len(results["results"]) <= 3
        assert "timestamp" in results
        assert len(results["full_documents"]) <= 3
    
    def test_strategy_b_retrieval(self, rag_pipeline):
        """Test Strategy B (AI-enhanced retrieval)"""
        query = "How does the system handle peak load?"
        
        results = rag_pipeline.retrieve_strategy_b(query, top_k=3)
        
        assert results["strategy"] == "B_AI_ENHANCED_RETRIEVAL"
        assert results["original_query"] == query
        assert "expanded_query" in results
        assert len(results["results"]) <= 3
        assert "expansion_info" in results
    
    def test_strategy_b_has_query_expansion(self, rag_pipeline):
        """Test that Strategy B includes query expansion"""
        query = "security"
        
        results = rag_pipeline.retrieve_strategy_b(query)
        
        # Expanded query should be different from original
        assert results["expanded_query"] != query
        assert "added_terms" in results["expansion_info"]
    
    def test_compare_strategies(self, rag_pipeline):
        """Test strategy comparison"""
        query = "How does the system handle peak load?"
        
        comparison = rag_pipeline.compare_strategies(query, top_k=3)
        
        assert comparison["query"] == query
        assert "strategy_a" in comparison
        assert "strategy_b" in comparison
        assert "metrics" in comparison
        
        metrics = comparison["metrics"]
        assert "strategy_a_avg_similarity" in metrics
        assert "strategy_b_avg_similarity" in metrics
        assert "improvement_percentage" in metrics
    
    def test_batch_compare(self, rag_pipeline):
        """Test batch comparison of strategies"""
        queries = [
            "How does the system handle peak load?",
            "What are the security mechanisms?",
            "Explain the architecture"
        ]
        
        comparisons = rag_pipeline.batch_compare(queries, top_k=3)
        
        assert len(comparisons) == len(queries)
        assert all("metrics" in c for c in comparisons)
    
    def test_results_have_valid_scores(self, rag_pipeline):
        """Test that similarity scores are valid"""
        query = "security"
        
        results_a = rag_pipeline.retrieve_strategy_a(query, top_k=3)
        results_b = rag_pipeline.retrieve_strategy_b(query, top_k=3)
        
        for result in results_a["results"]:
            score = result["similarity_score"]
            assert -1 <= score <= 1
        
        for result in results_b["results"]:
            score = result["similarity_score"]
            assert -1 <= score <= 1
    
    def test_pipeline_stats(self, rag_pipeline):
        """Test pipeline statistics"""
        stats = rag_pipeline.get_pipeline_stats()
        
        assert "embedding_model" in stats
        assert "vector_store" in stats
        assert stats["documents_ingested"] is True
        assert stats["query_expansion_enabled"] is True
    
    def test_document_chunking(self, rag_pipeline):
        """Test document chunking functionality"""
        long_doc = "Sentence one. Sentence two. " * 100
        
        chunks = RAGPipeline._chunk_documents([long_doc], chunk_size=100)
        
        assert len(chunks) > 1
        assert all(len(c) <= 150 for c in chunks)  # Some tolerance
    
    def test_retrieval_without_ingestion_fails(self):
        """Test that retrieval fails if documents not ingested"""
        rag = RAGPipeline()
        
        with pytest.raises(ValueError):
            rag.retrieve_strategy_a("test query")
        
        with pytest.raises(ValueError):
            rag.retrieve_strategy_b("test query")
    
    def test_semantic_relevance(self, rag_pipeline):
        """Test that retrieved results are semantically relevant"""
        # Query about load handling should retrieve the relevant document
        query = "peak load"
        
        results = rag_pipeline.retrieve_strategy_a(query, top_k=5)
        
        # Should have results
        assert len(results["results"]) > 0
        
        # Top result should mention peak/load/traffic concepts
        top_doc = results["full_documents"][0].lower()
        assert any(term in top_doc for term in ["load", "peak", "traffic", "scaling"])
