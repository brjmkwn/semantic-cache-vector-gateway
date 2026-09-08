from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os

class Settings(BaseSettings):
    APP_NAME: str = "Qdrant Enterprise Vector Search Demo"
    DEBUG: bool = True
    
    # OpenAI Settings (3072-dim flagship model)
    OPENAI_API_KEY: Optional[str] = None
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    EMBEDDING_DIMENSION: int = 3072
    
    # Qdrant Settings
    QDRANT_URL: Optional[str] = None
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_STORAGE_PATH: Optional[str] = "./qdrant_storage"
    QDRANT_COLLECTION_NAME: str = "enterprise_knowledge_base"
    
    # Vector Search Parameters
    DEFAULT_TOP_K: int = 5
    DEFAULT_SCORE_THRESHOLD: float = 0.35
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
