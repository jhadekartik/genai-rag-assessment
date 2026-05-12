"""Unit tests for embedding model"""

import pytest
import numpy as np
from src.embeddings import EmbeddingModel


@pytest.fixture
def embedding_model():
    """Fixture for embedding model"""
    return EmbeddingModel(model_name="all-MiniLM-L6-v2")


class TestEmbeddingModel:
    """Test cases for EmbeddingModel"""
    
    def test_initialization(self, embedding_model):
        """Test model initialization"""
        assert embedding_model.model_name == "all-MiniLM-L6-v2"
        assert embedding_model.embedding_dim == 384
    
    def test_embed_single_text(self, embedding_model):
        """Test embedding a single text"""
        text = "This is a test sentence"
        embedding = embedding_model.embed_text(text)
        
        assert embedding.shape == (1, 384)
        assert isinstance(embedding, np.ndarray)
    
    def test_embed_multiple_texts(self, embedding_model):
        """Test embedding multiple texts"""
        texts = [
            "This is the first sentence",
            "This is the second sentence",
            "This is the third sentence"
        ]
        embeddings = embedding_model.embed_text(texts)
        
        assert embeddings.shape == (3, 384)
        assert isinstance(embeddings, np.ndarray)
    
    def test_embed_documents(self, embedding_model):
        """Test embedding documents"""
        docs = [
            "Document about security",
            "Document about performance",
            "Document about scalability"
        ]
        embeddings = embedding_model.embed_documents(docs)
        
        assert embeddings.shape == (3, 384)
    
    def test_cosine_similarity(self, embedding_model):
        """Test cosine similarity calculation"""
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([1, 0, 0])
        
        similarity = EmbeddingModel.cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(1.0, abs=0.01)
    
    def test_cosine_similarity_orthogonal(self, embedding_model):
        """Test cosine similarity for orthogonal vectors"""
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([0, 1, 0])
        
        similarity = EmbeddingModel.cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(0, abs=0.01)
    
    def test_cosine_distance(self, embedding_model):
        """Test cosine distance calculation"""
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([1, 0, 0])
        
        distance = EmbeddingModel.cosine_distance(vec1, vec2)
        assert distance == pytest.approx(0.0, abs=0.01)
    
    def test_euclidean_distance(self, embedding_model):
        """Test euclidean distance calculation"""
        vec1 = np.array([0, 0, 0])
        vec2 = np.array([3, 4, 0])
        
        distance = EmbeddingModel.euclidean_distance(vec1, vec2)
        assert distance == pytest.approx(5.0, abs=0.01)
    
    def test_semantic_similarity(self, embedding_model):
        """Test semantic similarity between related sentences"""
        sent1 = "The cat is sleeping"
        sent2 = "The dog is sleeping"
        sent3 = "The sky is blue"
        
        emb1 = embedding_model.embed_text(sent1)[0]
        emb2 = embedding_model.embed_text(sent2)[0]
        emb3 = embedding_model.embed_text(sent3)[0]
        
        # Similar sentences should have higher similarity
        sim_12 = EmbeddingModel.cosine_similarity(emb1, emb2)
        sim_13 = EmbeddingModel.cosine_similarity(emb1, emb3)
        
        assert sim_12 > sim_13  # Sentences about animals should be more similar
    
    def test_model_info(self, embedding_model):
        """Test model info retrieval"""
        info = embedding_model.get_model_info()
        
        assert info["model_name"] == "all-MiniLM-L6-v2"
        assert info["embedding_dimension"] == 384
        assert "sentence-transformers" in info["model_type"]
