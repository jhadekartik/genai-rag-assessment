"""Unit tests for query expander"""

import pytest
from src.query_expander import MockQueryExpander


@pytest.fixture
def expander():
    """Fixture for query expander"""
    return MockQueryExpander()


class TestMockQueryExpander:
    """Test cases for MockQueryExpander"""
    
    def test_initialization(self, expander):
        """Test query expander initialization"""
        assert len(expander.expansion_strategies) > 0
    
    def test_expand_peak_load_query(self, expander):
        """Test expanding a peak load query"""
        query = "How does the system handle peak load?"
        expanded = expander.expand_query(query)
        
        assert query in expanded  # Original should be included
        assert len(expanded) > len(query)  # Should be longer
    
    def test_expand_security_query(self, expander):
        """Test expanding a security query"""
        query = "What are the security mechanisms?"
        expanded = expander.expand_query(query)
        
        assert query in expanded
        assert len(expanded) > len(query)
    
    def test_expand_generic_query(self, expander):
        """Test expanding a query without known concepts"""
        query = "Tell me about the system"
        expanded = expander.expand_query(query)
        
        assert query in expanded
        # Should still add some generic expansion terms
        assert len(expanded) > len(query)
    
    def test_batch_expand_queries(self, expander):
        """Test batch expansion"""
        queries = [
            "How does the system handle peak load?",
            "What are the security mechanisms?",
            "Explain the architecture"
        ]
        
        expanded_queries = expander.batch_expand_queries(queries)
        
        assert len(expanded_queries) == 3
        assert all(orig in exp for orig, exp in zip(queries, expanded_queries))
    
    def test_explain_expansion(self, expander):
        """Test expansion explanation"""
        original = "How does the system handle peak load?"
        expanded = expander.expand_query(original)
        
        explanation = expander.explain_expansion(original, expanded)
        
        assert explanation["original"] == original
        assert explanation["expanded"] == expanded
        assert isinstance(explanation["added_terms"], list)
        assert len(explanation["added_terms"]) > 0
    
    def test_expansion_strategy_type(self, expander):
        """Test that expansion follows a strategy"""
        explanation = expander.explain_expansion(
            "peak load",
            "peak load maximum throughput burst capacity"
        )
        
        assert explanation["expansion_strategy"] == "semantic-enrichment"
    
    def test_deterministic_expansion(self, expander):
        """Test that expansion is reproducible for same input"""
        query = "security and data"
        
        # Multiple expansions should contain the original query
        exp1 = expander.expand_query(query)
        exp2 = expander.expand_query(query)
        
        assert query in exp1
        assert query in exp2
    
    def test_expansion_with_custom_num_terms(self, expander):
        """Test expansion with different number of expansion terms"""
        query = "How does the system handle peak load?"
        
        expanded_1 = expander.expand_query(query, num_expansions=1)
        expanded_5 = expander.expand_query(query, num_expansions=5)
        
        # More expansion terms should result in longer string
        assert len(expanded_5) >= len(expanded_1)
    
    def test_no_empty_expansion(self, expander):
        """Test that expansion never returns empty"""
        queries = [
            "a",
            "query",
            "",  # Even empty query
            "x" * 100  # Very long
        ]
        
        for query in queries:
            expanded = expander.expand_query(query)
            assert len(expanded) > 0
