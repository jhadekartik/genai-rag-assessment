"""Unit tests for vector store"""

import pytest
import numpy as np
from src.vector_store import FAISSVectorStore


@pytest.fixture
def vector_store():
    """Fixture for vector store"""
    return FAISSVectorStore(embedding_dim=384)


@pytest.fixture
def sample_docs_and_embeddings():
    """Fixture providing sample documents and embeddings"""
    docs = [
        "This is about system security and data protection",
        "Scalability requires load balancing and auto-scaling",
        "Performance optimization uses caching strategies",
        "Database indexing improves query efficiency",
        "API rate limiting prevents abuse"
    ]
    
    # Create synthetic embeddings (normalized random vectors)
    embeddings = np.random.randn(5, 384)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    return docs, embeddings.astype(np.float32)


class TestFAISSVectorStore:
    """Test cases for FAISSVectorStore"""
    
    def test_initialization(self, vector_store):
        """Test vector store initialization"""
        assert vector_store.embedding_dim == 384
        assert vector_store.document_count == 0
    
    def test_add_documents(self, vector_store, sample_docs_and_embeddings):
        """Test adding documents to store"""
        docs, embeddings = sample_docs_and_embeddings
        
        vector_store.add_documents(docs, embeddings)
        
        assert vector_store.document_count == 5
        assert len(vector_store.documents) == 5
    
    def test_dimension_mismatch(self, vector_store, sample_docs_and_embeddings):
        """Test error handling for dimension mismatch"""
        docs, _ = sample_docs_and_embeddings
        wrong_embeddings = np.random.randn(5, 256)  # Wrong dimension
        
        with pytest.raises(ValueError):
            vector_store.add_documents(docs, wrong_embeddings)
    
    def test_count_mismatch(self, vector_store, sample_docs_and_embeddings):
        """Test error handling for document/embedding count mismatch"""
        docs, embeddings = sample_docs_and_embeddings
        
        with pytest.raises(ValueError):
            vector_store.add_documents(docs[:3], embeddings)  # 3 docs, 5 embeddings
    
    def test_search(self, vector_store, sample_docs_and_embeddings):
        """Test searching in vector store"""
        docs, embeddings = sample_docs_and_embeddings
        vector_store.add_documents(docs, embeddings)
        
        # Search with first document's embedding
        query_embedding = embeddings[0]
        results = vector_store.search(query_embedding, top_k=3)
        
        assert len(results) <= 3
        assert all(isinstance(r, tuple) and len(r) == 3 for r in results)
        
        # First result should be the document itself
        assert results[0][0] == 0  # Document ID
    
    def test_batch_search(self, vector_store, sample_docs_and_embeddings):
        """Test batch searching"""
        docs, embeddings = sample_docs_and_embeddings
        vector_store.add_documents(docs, embeddings)
        
        # Create query embeddings
        query_embeddings = embeddings[:2]  # Use first 2 as queries
        
        batch_results = vector_store.batch_search(query_embeddings, top_k=2)
        
        assert len(batch_results) == 2
        assert all(len(r) <= 2 for r in batch_results)
    
    def test_get_document(self, vector_store, sample_docs_and_embeddings):
        """Test document retrieval by ID"""
        docs, embeddings = sample_docs_and_embeddings
        vector_store.add_documents(docs, embeddings)
        
        retrieved = vector_store.get_document(0)
        assert retrieved == docs[0]
    
    def test_get_document_invalid_id(self, vector_store, sample_docs_and_embeddings):
        """Test error handling for invalid document ID"""
        docs, embeddings = sample_docs_and_embeddings
        vector_store.add_documents(docs, embeddings)
        
        with pytest.raises(IndexError):
            vector_store.get_document(999)
    
    def test_similarity_scores_range(self, vector_store, sample_docs_and_embeddings):
        """Test that similarity scores are in expected range"""
        docs, embeddings = sample_docs_and_embeddings
        vector_store.add_documents(docs, embeddings)
        
        results = vector_store.search(embeddings[0], top_k=5)
        
        # Cosine similarity should be between -1 and 1
        for _, similarity, _ in results:
            assert -1 <= similarity <= 1
    
    def test_get_stats(self, vector_store, sample_docs_and_embeddings):
        """Test statistics retrieval"""
        docs, embeddings = sample_docs_and_embeddings
        vector_store.add_documents(docs, embeddings)
        
        stats = vector_store.get_stats()
        
        assert stats["document_count"] == 5
        assert stats["embedding_dimension"] == 384
        assert stats["similarity_metric"] == "cosine"
        assert stats["memory_usage_mb"] > 0
    
    def test_empty_search(self, vector_store):
        """Test searching on empty store"""
        query = np.random.randn(384)
        results = vector_store.search(query, top_k=3)
        
        assert len(results) == 0
