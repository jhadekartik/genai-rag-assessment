"""Query expansion using mocked LLM for improved retrieval"""

import logging
from typing import List
import random

logger = logging.getLogger(__name__)


class MockQueryExpander:
    """
    Mocked query expansion model simulating Vertex AI's GenerativeModel.
    Expands queries to improve semantic search by:
    1. Adding related terminology
    2. Paraphrasing concepts
    3. Including alternative phrasings
    """
    
    def __init__(self):
        """Initialize the mock query expander"""
        self.expansion_strategies = {
            "peak load": [
                "maximum throughput", "high traffic", "burst capacity",
                "scalability under stress", "load balancing"
            ],
            "security": [
                "data protection", "encryption", "authentication",
                "access control", "compliance", "vulnerability"
            ],
            "architecture": [
                "system design", "component structure", "microservices",
                "deployment topology", "infrastructure"
            ],
            "scalability": [
                "horizontal scaling", "vertical scaling", "distributed",
                "performance", "resource utilization"
            ],
            "data": [
                "information", "storage", "database", "processing",
                "retrieval", "consistency"
            ]
        }
        logger.info("Initialized MockQueryExpander")
    
    def expand_query(self, query: str, num_expansions: int = 3) -> str:
        """
        Expand a query by adding related concepts and alternative phrasings.
        
        Args:
            query: Original query string
            num_expansions: Number of alternative terms to add
            
        Returns:
            Expanded query string
        """
        expanded_terms = [query]
        
        # Find matching expansion strategies
        query_lower = query.lower()
        matched_concepts = []
        
        for concept, related_terms in self.expansion_strategies.items():
            if concept in query_lower:
                matched_concepts.extend(related_terms)
        
        # If no direct matches, use general expansion
        if not matched_concepts:
            matched_concepts = self._generate_generic_expansion(query)
        
        # Randomly select expansion terms
        if matched_concepts:
            selected_expansions = random.sample(
                matched_concepts,
                min(num_expansions, len(matched_concepts))
            )
            expanded_terms.extend(selected_expansions)
        
        expanded_query = " ".join(expanded_terms)
        logger.info(f"Expanded query: '{query}' -> '{expanded_query}'")
        
        return expanded_query
    
    def _generate_generic_expansion(self, query: str) -> List[str]:
        """Generate generic expansion terms for unknown queries"""
        generic_terms = [
            "approach", "method", "implementation",
            "process", "solution", "mechanism",
            "strategy", "design", "framework"
        ]
        return generic_terms
    
    def batch_expand_queries(self, queries: List[str], num_expansions: int = 3) -> List[str]:
        """
        Expand multiple queries.
        
        Args:
            queries: List of query strings
            num_expansions: Number of expansion terms per query
            
        Returns:
            List of expanded queries
        """
        expanded_queries = [self.expand_query(q, num_expansions) for q in queries]
        logger.info(f"Expanded {len(queries)} queries")
        return expanded_queries
    
    def explain_expansion(self, original_query: str, expanded_query: str) -> dict:
        """
        Explain what terms were added during expansion.
        
        Args:
            original_query: Original query
            expanded_query: Expanded query
            
        Returns:
            Dictionary with explanation
        """
        original_tokens = set(original_query.lower().split())
        expanded_tokens = set(expanded_query.lower().split())
        
        added_terms = expanded_tokens - original_tokens
        
        return {
            "original": original_query,
            "expanded": expanded_query,
            "added_terms": list(added_terms),
            "expansion_strategy": "semantic-enrichment"
        }
