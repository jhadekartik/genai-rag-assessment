# Semantic RAG & Vector Search Assessment

**Focus**: Embeddings, Vector Databases, Retrieval Logic, and Benchmarking

This project implements a semantic retrieval pipeline using sentence-transformers and FAISS. It compares two retrieval strategies: direct embedding search vs. query-expanded search.

## Overview

The project demonstrates:
- **Embedding Generation**: Using sentence-transformers (local equivalent to Vertex AI's text-embedding-gecko)
- **Vector Storage**: FAISS-based vector database for semantic search
- **Retrieval Strategies**: Two approaches with quantitative comparison
- **Query Expansion**: Simple term enrichment for improved retrieval

## Project Structure

```
semantic-rag-assessment/
├── src/                        # Source code
│   ├── __init__.py
│   ├── embeddings.py           # Embedding model wrapper
│   ├── vector_store.py         # FAISS-based vector database
│   ├── retriever.py            # Core retrieval orchestration
│   ├── query_expander.py       # Query expansion logic
│   └── config.py               # Configuration
├── tests/                      # Test suite (41 tests)
│   ├── __init__.py
│   ├── test_embeddings.py      # Embedding tests
│   ├── test_vector_store.py    # Vector store tests
│   ├── test_retriever.py       # Integration tests
│   └── test_query_expander.py  # Query expansion tests
├── data/
│   └── sample_documents.txt     # Sample dataset
├── README.md
├── IMPLEMENTATION.md            # Technical documentation
├── benchmark.py                 # Benchmark script
├── benchmark_results.json       # Results
├── retrieval_benchmark.md       # Analysis report
├── requirements.txt
├── pytest.ini
└── .gitignore
```

## Key Features

### Strategy A: Raw Vector Search
Direct embedding-based similarity search (baseline).

### Strategy B: Query Expansion  
Query expansion using mocked LLM to enrich queries before embedding and search.

## About This Project

Built as part of a GenAI semantic retrieval assessment, focusing on embeddings, vector databases, and comparing retrieval strategies with real performance metrics.

## Technical Decisions

- **sentence-transformers**: BERT-based sentence embeddings
- **FAISS**: Approximate nearest neighbor search
- **Mock Vertex AI**: Simulates GCP equivalents without cloud dependencies
- **Cosine Distance**: Standard metric for NLP embeddings

## Usage

```python
from src.retriever import RAGPipeline

# Initialize pipeline
rag = RAGPipeline(model_name="all-MiniLM-L6-v2", vector_dim=384)

# Ingest documents
documents = ["Document 1", "Document 2", ...]
rag.ingest_documents(documents)

# Strategy A: Direct retrieval
results_a = rag.retrieve_strategy_a("query", top_k=3)

# Strategy B: Query-expanded retrieval
results_b = rag.retrieve_strategy_b("query", top_k=3)
```

## Running Tests

```bash
pytest tests/ -v
```

## Running Benchmark

```bash
python benchmark.py
```

Results are saved to `benchmark_results.json` and `retrieval_benchmark.md`.
