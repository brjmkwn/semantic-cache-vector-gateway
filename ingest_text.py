import os
import sys
import time

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from core.document_loader import TextDocumentLoader
from core.qdrant_manager import QdrantManager
from core.embeddings import EmbeddingService
from core.search_engine import VectorSearchEngine
from config.settings import settings

def main():
    print("=" * 80)
    print("  📁 QDRANT CUSTOM TEXT INGESTION PIPELINE")
    print("=" * 80)
    
    inputs_dir = os.path.join(os.path.dirname(__file__), "data", "inputs")
    print(f"Scanning for text files in: {inputs_dir}")
    
    loader = TextDocumentLoader(chunk_size=500, chunk_overlap=80)
    documents = loader.load_directory(inputs_dir)
    
    if not documents:
        print(f"\n[!] No text files found in {inputs_dir}.")
        print("    Drop your .txt, .md, or .csv files into 'data/inputs/' and run this script again!")
        return

    print(f"\nTotal text chunks ready for vector indexing: {len(documents)}")
    
    # Initialize Qdrant and Ingest
    qdrant_mgr = QdrantManager()
    embedder = EmbeddingService()
    engine = VectorSearchEngine(qdrant_mgr=qdrant_mgr, embedding_svc=embedder)
    
    t0 = time.perf_counter()
    count = engine.index_documents(documents)
    total_time = (time.perf_counter() - t0) * 1000
    
    info = qdrant_mgr.get_collection_info()
    print("\n" + "=" * 80)
    print("  ✅ INGESTION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Indexed Chunks:         {count}")
    print(f"Total Points in Qdrant: {info['points_count']}")
    print(f"Vector Dimension:       {info['vector_dimension']} (text-embedding-3-large)")
    print(f"Total Processing Time:  {total_time:.2f}ms")
    print("\nTry searching your custom data using the FastAPI API: python api/main.py")
    qdrant_mgr.close()

if __name__ == "__main__":
    main()
