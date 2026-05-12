"""Configuration constants for RAG pipeline"""

import os
from typing import Dict

# Embedding Model Configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Lightweight, fast model
EMBEDDING_DIMENSION = 384  # Output dimension for all-MiniLM-L6-v2

# Vector Database Configuration
VECTOR_DB_TYPE = "FAISS"  # Using FAISS for local implementation
BATCH_SIZE_INGEST = 32
MAX_SEQUENCE_LENGTH = 512

# Retrieval Configuration
DEFAULT_TOP_K = 3
SIMILARITY_METRIC = "cosine"  # Options: cosine, euclidean

# Query Expansion Configuration
ENABLE_QUERY_EXPANSION = True
QUERY_EXPANSION_PROMPT_TEMPLATE = """You are a query optimization expert. 
Rewrite the following query to be more semantically rich and search-friendly, 
expanding on key concepts and alternative phrasings:

Original Query: {query}

Expanded Query:"""

# Similarity Thresholds
MIN_SIMILARITY_THRESHOLD = 0.3  # Minimum confidence for retrieval
OPTIMAL_SIMILARITY_THRESHOLD = 0.7  # Good match threshold

# Data Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAMPLE_DOCUMENTS_PATH = os.path.join(DATA_DIR, "sample_documents.txt")

# Benchmark Configuration
BENCHMARK_QUERIES = [
    "How does the system handle peak load?",
    "What are the security mechanisms for data protection?",
    "Explain the architecture and scalability approach",
]

BENCHMARK_OUTPUT_FORMAT = "json"  # Options: json, markdown, table
