import time
import uuid
from typing import List, Dict, Any, Optional
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchValue, Range
from core.qdrant_manager import QdrantManager
from core.embeddings import EmbeddingService
from config.settings import settings

def _ensure_valid_qdrant_id(raw_id: Any) -> str:
    """Ensures point ID is a valid UUID string compliant with Qdrant specs."""
    if isinstance(raw_id, int):
        return raw_id
    if isinstance(raw_id, str):
        try:
            uuid.UUID(raw_id)
            return raw_id
        except ValueError:
            return str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_id))
    return str(uuid.uuid4())

class VectorSearchEngine:
    """
    High-level Vector Search and Ingestion Engine.
    Handles embedding generation, batch upsert, metadata payload indexing, and filtered semantic queries.
    """
    def __init__(self, qdrant_mgr: QdrantManager = None, embedding_svc: EmbeddingService = None):
        self.qdrant = qdrant_mgr or QdrantManager()
        self.embedder = embedding_svc or EmbeddingService()
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        
        # Ensure collection is ready
        self.qdrant.create_collection_if_not_exists(self.collection_name)

    def index_documents(self, documents: List[Dict[str, Any]], batch_size: int = 32) -> int:
        """
        Converts document texts to 3072-dim embeddings and indexes them into Qdrant in resilient batches.
        """
        texts = [doc["text"] for doc in documents]
        print(f"[SearchEngine] Generating 3072-dim embeddings for {len(texts)} documents (Live: {self.embedder.is_live})...")
        start_t = time.perf_counter()
        
        # Process embeddings in batches of 50 for API resilience
        vectors = []
        for b_idx in range(0, len(texts), 50):
            sub_texts = texts[b_idx:b_idx+50]
            sub_vectors = self.embedder.get_embeddings_batch(sub_texts)
            vectors.extend(sub_vectors)
            print(f"  -> Generated {len(vectors)}/{len(texts)} embeddings...")
            
        emb_time = (time.perf_counter() - start_t) * 1000
        print(f"[SearchEngine] All embeddings generated in {emb_time:.2f}ms.")
        
        points = []
        for i, (doc, vector) in enumerate(zip(documents, vectors)):
            raw_id = doc.get("id") or f"doc_{i}_{doc.get('title', '')}"
            point_id = _ensure_valid_qdrant_id(raw_id)
            
            payload = {
                "original_id": str(raw_id),
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
                "category": doc.get("category", "general"),
                "author": doc.get("author", "anonymous"),
                "department": doc.get("department", "engineering"),
                "year": doc.get("year", 2026),
                "page_number": doc.get("page_number"),
                "file_name": doc.get("file_name"),
                "chunk_index": doc.get("chunk_index"),
                "total_chunks": doc.get("total_chunks"),
                "tags": doc.get("tags", [])
            }
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))
            
        print(f"[SearchEngine] Upserting {len(points)} points into Qdrant in batches of {batch_size}...")
        total_upserted = 0
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.qdrant.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=True
            )
            total_upserted += len(batch)
            print(f"  -> Upserted batch {total_upserted}/{len(points)} points to Qdrant.")
            
        print(f"[SearchEngine] Upsert complete. Total points indexed: {total_upserted}")
        return total_upserted

    def search(
        self,
        query: str,
        limit: int = 5,
        score_threshold: Optional[float] = None,
        category: Optional[str] = None,
        department: Optional[str] = None,
        min_year: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes a semantic vector similarity search with optional payload pre-filtering.
        """
        # 1. Generate query vector (3072 dims)
        t0 = time.perf_counter()
        query_vector = self.embedder.get_embedding(query)
        t_emb = (time.perf_counter() - t0) * 1000
        
        # 2. Build structured payload filters (executed directly inside HNSW graph)
        must_conditions = []
        if category:
            must_conditions.append(FieldCondition(key="category", match=MatchValue(value=category)))
        if department:
            must_conditions.append(FieldCondition(key="department", match=MatchValue(value=department)))
        if min_year:
            must_conditions.append(FieldCondition(key="year", range=Range(gte=min_year)))
            
        query_filter = Filter(must=must_conditions) if must_conditions else None
        
        # 3. Query Qdrant via modern query_points API
        t1 = time.perf_counter()
        query_kwargs = {
            "collection_name": self.collection_name,
            "query": query_vector,
            "query_filter": query_filter,
            "limit": limit
        }
        if score_threshold is not None:
            query_kwargs["score_threshold"] = score_threshold
            
        response = self.qdrant.client.query_points(**query_kwargs)
        t_search = (time.perf_counter() - t1) * 1000
        
        # Format results
        output = []
        for hit in response.points:
            p = hit.payload or {}
            output.append({
                "id": str(hit.id),
                "original_id": p.get("original_id"),
                "score": round(hit.score, 4),
                "title": p.get("title", ""),
                "text": p.get("text", ""),
                "category": p.get("category", ""),
                "department": p.get("department", ""),
                "year": p.get("year", 2026),
                "page_number": p.get("page_number"),
                "file_name": p.get("file_name"),
                "tags": p.get("tags", []),
                "metrics": {
                    "embedding_latency_ms": round(t_emb, 2),
                    "search_latency_ms": round(t_search, 2)
                }
            })
            
        return output
