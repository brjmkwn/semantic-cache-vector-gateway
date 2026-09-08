import os
import hashlib
import numpy as np
import re
from typing import List
from config.settings import settings

class EmbeddingService:
    """
    Handles vector embedding generation using OpenAI text-embedding-3-large (3072 dimensions).
    Includes an automatic deterministic offline fallback for local demo runs without an active API key.
    """
    def __init__(self, api_key: str = None, model: str = None, dimension: int = None):
        self.api_key = api_key or settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        self.model = model or settings.EMBEDDING_MODEL
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        self._client = None
        
        if self.api_key and self.api_key.startswith("sk-"):
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except Exception as e:
                print(f"[Warning] Failed to initialize OpenAI client: {e}. Falling back to offline embedding generator.")

    @property
    def is_live(self) -> bool:
        return self._client is not None

    def get_embedding(self, text: str) -> List[float]:
        """Generate a 3072-dimensional embedding vector for a single text."""
        if self.is_live:
            try:
                response = self._client.embeddings.create(
                    input=[text],
                    model=self.model,
                    dimensions=self.dimension
                )
                return response.data[0].embedding
            except Exception as e:
                print(f"[OpenAI API Error] {e}. Falling back to offline deterministic vector.")
        
        return self._generate_deterministic_vector(text)

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate 3072-dimensional embeddings for a batch of texts."""
        if self.is_live:
            try:
                response = self._client.embeddings.create(
                    input=texts,
                    model=self.model,
                    dimensions=self.dimension
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                print(f"[OpenAI API Error] {e}. Falling back to offline deterministic vectors.")
        
        return [self._generate_deterministic_vector(t) for t in texts]

    def _generate_deterministic_vector(self, text: str) -> List[float]:
        """
        Generates a high-entropy, unit-normalized 3072-dimensional pseudo-embedding vector 
        derived deterministically using subword n-gram semantic feature hashing.
        Allows realistic semantic similarity search in offline/demo mode without API costs.
        """
        # 1. Base vector initialized with document-level hash
        doc_seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(doc_seed)
        vector = rng.standard_normal(self.dimension) * 0.1
        
        # 2. Extract words and 3-to-5 character n-grams for semantic fuzzy matching
        clean_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
        tokens = clean_text.split()
        
        features = set()
        for token in tokens:
            features.add(token[:4])
            features.add(token[:6])
            features.add(token)
            if len(token) >= 3:
                for j in range(len(token) - 2):
                    features.add(token[j:j+3])
                    
        for feat in features:
            f_seed = int(hashlib.md5(feat.encode("utf-8")).hexdigest()[:8], 16)
            f_rng = np.random.default_rng(f_seed)
            vector += f_rng.standard_normal(self.dimension) * 1.0
            
        # 3. L2-normalize to unit length (standard for Cosine Distance)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector.tolist()
