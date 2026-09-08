import json
import os
import shutil
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from pydantic import BaseModel, Field
from core.qdrant_manager import QdrantManager
from core.embeddings import EmbeddingService
from core.search_engine import VectorSearchEngine
from core.document_loader import TextDocumentLoader
from config.settings import settings

app = FastAPI(
    title="Qdrant Enterprise Vector Search API",
    description="Production-ready Vector Search Service with Custom Text Ingestion using Qdrant and 3072-dim text-embedding-3-large embeddings.",
    version="1.1.0"
)

# Initialize singletons
qdrant_mgr = QdrantManager()
embedder = EmbeddingService()
engine = VectorSearchEngine(qdrant_mgr=qdrant_mgr, embedding_svc=embedder)
loader = TextDocumentLoader(chunk_size=500, chunk_overlap=80)

class SearchRequest(BaseModel):
    query: str = Field(..., example="What are the API key security rules and GDPR compliance policies?")
    limit: int = Field(default=5, ge=1, le=50)
    score_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Minimum cosine similarity score threshold (optional)")
    category: Optional[str] = Field(default=None, example="imported_files")
    department: Optional[str] = Field(default=None, example="enterprise")
    min_year: Optional[int] = Field(default=None, example=2025)

class DocumentItem(BaseModel):
    id: Optional[str] = None
    title: str
    text: str
    category: str = "general"
    department: str = "engineering"
    year: int = 2026
    tags: List[str] = []

@app.get("/", tags=["Health"])
def root():
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "engine": "Qdrant Vector Database",
        "embedding_model": settings.EMBEDDING_MODEL,
        "vector_dimension": settings.EMBEDDING_DIMENSION,
        "qdrant_mode": qdrant_mgr.mode,
        "live_openai_embeddings": embedder.is_live
    }

@app.get("/collection/info", tags=["Collection"])
def collection_info():
    try:
        return qdrant_mgr.get_collection_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/upsert-raw-json", tags=["Ingestion"])
def upsert_documents(documents: List[DocumentItem]):
    """Upsert pre-structured JSON documents into Qdrant."""
    try:
        doc_dicts = [d.dict() for d in documents]
        count = engine.index_documents(doc_dicts)
        return {"status": "success", "indexed_count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/upload-text-file", tags=["Ingestion"])
async def upload_text_file(
    file: UploadFile = File(..., description="Upload a .txt, .md, or .csv document"),
    category: str = Form(default="custom_upload"),
    department: str = Form(default="enterprise")
):
    """
    Upload your own text file (.txt, .md, .csv), automatically split into semantic chunks, 
    generate 3072-dim embeddings, and index into Qdrant!
    """
    try:
        inputs_dir = os.path.join(os.path.dirname(__file__), "..", "data", "inputs")
        os.makedirs(inputs_dir, exist_ok=True)
        
        saved_path = os.path.join(inputs_dir, file.filename)
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        chunks = loader.load_and_chunk_file(saved_path, category=category, department=department)
        if not chunks:
            raise HTTPException(status_code=400, detail="Uploaded file was empty.")
            
        count = engine.index_documents(chunks)
        return {
            "status": "success",
            "file_name": file.filename,
            "chunks_created_and_indexed": count,
            "sample_chunk_title": chunks[0]["title"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/ingest-inputs-folder", tags=["Ingestion"])
def ingest_inputs_folder():
    """Scans 'data/inputs/' folder and indexes all text files into Qdrant."""
    try:
        inputs_dir = os.path.join(os.path.dirname(__file__), "..", "data", "inputs")
        chunks = loader.load_directory(inputs_dir)
        if not chunks:
            return {"status": "warning", "message": "No files found in data/inputs/ folder."}
        
        count = engine.index_documents(chunks)
        return {"status": "success", "indexed_chunks": count, "message": f"Successfully indexed {count} chunks from data/inputs/ into Qdrant."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", tags=["Vector Search"])
def vector_search(req: SearchRequest):
    """Execute 3072-dim semantic similarity search with metadata pre-filtering."""
    try:
        results = engine.search(
            query=req.query,
            limit=req.limit,
            score_threshold=req.score_threshold,
            category=req.category,
            department=req.department,
            min_year=req.min_year
        )
        return {
            "query": req.query,
            "results_count": len(results),
            "filters_applied": {
                "category": req.category,
                "department": req.department,
                "min_year": req.min_year
            },
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
