import os
from typing import Optional, Dict, Any, List
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, HnswConfigDiff
from config.settings import settings

class QdrantManager:
    """
    Enterprise Qdrant Client and Collection Manager.
    Supports Qdrant Cloud clusters, remote Docker endpoints, or embedded local/in-memory storage.
    Configured specifically for 3072-dimensional text-embedding-3-large vectors.
    """
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        storage_path: Optional[str] = None,
        in_memory: bool = False,
        timeout: float = 60.0
    ):
        self.url = url or settings.QDRANT_URL or os.getenv("QDRANT_URL")
        self.api_key = api_key or settings.QDRANT_API_KEY or os.getenv("QDRANT_API_KEY")
        self.storage_path = storage_path or settings.QDRANT_STORAGE_PATH
        self.in_memory = in_memory
        self.timeout = timeout
        
        if self.in_memory:
            self.client = QdrantClient(":memory:", timeout=self.timeout)
            self.mode = "memory"
        elif self.url:
            print(f"[QdrantManager] Connecting to Qdrant Cloud / Remote Server: {self.url}")
            self.client = QdrantClient(url=self.url, api_key=self.api_key, timeout=self.timeout)
            self.mode = "cloud"
        else:
            abs_storage = os.path.abspath(self.storage_path)
            os.makedirs(abs_storage, exist_ok=True)
            print(f"[QdrantManager] Initializing Local Persistent Qdrant Engine at: {abs_storage}")
            self.client = QdrantClient(path=abs_storage, timeout=self.timeout)
            self.mode = "local"

    def create_collection_if_not_exists(
        self,
        collection_name: str = None,
        dimension: int = None,
        distance: Distance = Distance.COSINE
    ) -> bool:
        """
        Creates an enterprise collection with 3072 vector dimensions and optimized HNSW graph index.
        """
        collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        dim = dimension or settings.EMBEDDING_DIMENSION
        
        collections_response = self.client.get_collections()
        existing = [c.name for c in collections_response.collections]
        
        if collection_name in existing:
            return False
            
        print(f"[QdrantManager] Creating Collection '{collection_name}' with {dim} dimensions (Metric: {distance.name})...")
        
        on_disk_flag = not self.in_memory
        
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=dim,
                distance=distance,
                on_disk=on_disk_flag
            ),
            hnsw_config=HnswConfigDiff(
                m=16,
                ef_construct=100,
                full_scan_threshold=1000,
                on_disk=on_disk_flag
            )
        )
        print(f"[QdrantManager] Collection '{collection_name}' created successfully.")
        
        # Create payload indexes for pre-filtered vector queries
        self._create_payload_indexes(collection_name)
        return True

    def _create_payload_indexes(self, collection_name: str):
        """Creates keyword and numeric payload indexes for fast filtered vector searches."""
        keyword_fields = ["category", "department", "file_name", "tags"]
        integer_fields = ["year", "page_number"]
        
        for field in keyword_fields:
            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
            except Exception:
                pass
                
        for field in integer_fields:
            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.INTEGER
                )
            except Exception:
                pass

    def recreate_collection(
        self,
        collection_name: str = None,
        dimension: int = None,
        distance: Distance = Distance.COSINE
    ):
        """Drops and recreates the collection (useful for fresh ingest / resets)."""
        collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        dim = dimension or settings.EMBEDDING_DIMENSION
        
        collections_response = self.client.get_collections()
        existing = [c.name for c in collections_response.collections]
        
        if collection_name in existing:
            print(f"[QdrantManager] Deleting existing collection '{collection_name}'...")
            self.client.delete_collection(collection_name=collection_name)
            
        return self.create_collection_if_not_exists(collection_name, dim, distance)

    def get_collection_info(self, collection_name: str = None) -> Dict[str, Any]:
        """Returns detailed metadata and point count for the collection."""
        collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        info = self.client.get_collection(collection_name=collection_name)
        
        status_str = info.status.name if hasattr(info.status, 'name') else str(info.status)
        dist_str = info.config.params.vectors.distance.name if hasattr(info.config.params.vectors.distance, 'name') else str(info.config.params.vectors.distance)
        
        return {
            "name": collection_name,
            "status": status_str,
            "points_count": info.points_count or 0,
            "indexed_vectors_count": getattr(info, 'indexed_vectors_count', 0),
            "vector_dimension": info.config.params.vectors.size,
            "distance_metric": dist_str
        }

    def close(self):
        """Closes the Qdrant client connection releasing local file locks."""
        if hasattr(self.client, 'close'):
            self.client.close()
