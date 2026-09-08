import pytest
from core.qdrant_manager import QdrantManager
from core.embeddings import EmbeddingService
from core.search_engine import VectorSearchEngine
from config.settings import settings

def test_embedding_dimension():
    embedder = EmbeddingService()
    vector = embedder.get_embedding("Test enterprise search query")
    assert len(vector) == 3072, f"Expected 3072 dimensions, got {len(vector)}"

def test_qdrant_collection_lifecycle():
    qdrant_mgr = QdrantManager(in_memory=True)
    test_collection = "test_verification_collection"
    
    # Create
    qdrant_mgr.recreate_collection(test_collection, dimension=3072)
    info = qdrant_mgr.get_collection_info(test_collection)
    
    assert info["vector_dimension"] == 3072
    assert "COSINE" in info["distance_metric"].upper()
    qdrant_mgr.close()

def test_search_engine_flow():
    qdrant_mgr = QdrantManager(in_memory=True)
    embedder = EmbeddingService()
    engine = VectorSearchEngine(qdrant_mgr=qdrant_mgr, embedding_svc=embedder)
    
    docs = [
        {
            "id": "test_01",
            "title": "Cloud Security Policy",
            "text": "All API keys must be rotated every 90 days in HashiCorp Vault.",
            "category": "security",
            "department": "devsecops",
            "year": 2026,
            "tags": ["security", "vault"]
        },
        {
            "id": "test_02",
            "title": "FastAPI Web Architecture",
            "text": "Building high speed microservices with FastAPI and Pydantic.",
            "category": "architecture",
            "department": "engineering",
            "year": 2026,
            "tags": ["fastapi", "python"]
        }
    ]
    
    indexed = engine.index_documents(docs)
    assert indexed == 2
    
    # Unfiltered search
    results = engine.search(query="How to rotate secret tokens?", limit=1)
    assert len(results) > 0
    assert results[0]["title"] == "Cloud Security Policy"
    
    # Filtered search
    filtered = engine.search(query="API and web framework", category="architecture", limit=1)
    assert len(filtered) > 0
    assert filtered[0]["category"] == "architecture"
    qdrant_mgr.close()
