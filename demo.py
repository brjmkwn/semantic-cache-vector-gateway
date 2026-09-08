import os
import sys
import time

# Ensure UTF-8 output on Windows terminal
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from core.qdrant_manager import QdrantManager
from core.embeddings import EmbeddingService
from core.search_engine import VectorSearchEngine
from core.document_loader import TextDocumentLoader
from config.settings import settings

def print_banner(title: str):
    print("\n" + "=" * 85)
    print(f"  {title}")
    print("=" * 85)

def display_search_result(rank: int, r: dict):
    print(f"\n  [*] Result #{rank} [Cosine Similarity Score: {r['score']:.4f}]")
    print(f"     Source File: {r.get('file_name', 'N/A')} (Page: {r.get('page_number', 'N/A')})")
    print(f"     Section:     {r.get('title', 'N/A')}")
    print(f"     Snippet:     {r.get('text', '')[:250]}...")
    print(f"     Latency:     Embedding: {r['metrics']['embedding_latency_ms']}ms | Qdrant Search: {r['metrics']['search_latency_ms']}ms")

def run_enterprise_demo():
    print_banner(">> QDRANT ENTERPRISE CUSTOM VECTOR SEARCH DEMONSTRATION")
    print(f"Embedding Model:    {settings.EMBEDDING_MODEL} (3,072-Dimensional OpenAI Embeddings)")
    print(f"Distance Metric:    Cosine Similarity")
    print(f"Target Collection:  {settings.QDRANT_COLLECTION_NAME}")
    
    # 1. Initialize Engine
    print("\n[1/3] Connecting to Qdrant Cloud...")
    qdrant_mgr = QdrantManager()
    embedder = EmbeddingService()
    
    if embedder.is_live:
        print("  -> Live OpenAI API Key active: Real text-embedding-3-large embeddings.")
    else:
        print("  -> Fallback Mode active.")

    engine = VectorSearchEngine(qdrant_mgr=qdrant_mgr, embedding_svc=embedder)
    
    # Check collection status
    try:
        info = qdrant_mgr.get_collection_info()
    except Exception:
        qdrant_mgr.create_collection_if_not_exists()
        info = qdrant_mgr.get_collection_info()

    print(f"  Collection Status:      {info['status']}")
    print(f"  Total Points in Qdrant: {info['points_count']}")
    print(f"  Vector Dimension:       {info['vector_dimension']}")
    print(f"  Distance Metric:        {info['distance_metric']}")

    # If collection is empty, automatically ingest data/inputs
    if info['points_count'] == 0:
        print("\n[!] Collection is empty. Auto-ingesting custom files from data/inputs/...")
        inputs_dir = os.path.join(os.path.dirname(__file__), "data", "inputs")
        loader = TextDocumentLoader(chunk_size=500, chunk_overlap=80)
        docs = loader.load_directory(inputs_dir)
        if docs:
            engine.index_documents(docs)
            info = qdrant_mgr.get_collection_info()
            print(f"  -> Ingested {info['points_count']} chunks into Qdrant Cloud!")

    # 2. Automated Sample Queries on Custom Data
    print_banner("[2/3] RUNNING SAMPLE SEMANTIC SEARCH QUERIES ON YOUR CUSTOM DATA")
    
    sample_queries = [
        {
            "prompt": "What is the strategy for fighting multiple enemies at once?",
            "desc": "Strategy on multiple opponents (The Book of Five Rings)"
        },
        {
            "prompt": "What is the attitude of the mind according to the Water Book?",
            "desc": "Mindset & Water Chapter principles (The Book of Five Rings)"
        },
        {
            "prompt": "What is our company policy on AI governance, model evaluation, and safety?",
            "desc": "Enterprise Policy Query (AI Governance Policy)"
        }
    ]

    for i, q in enumerate(sample_queries, 1):
        print(f"\n--- Query #{i}: \"{q['prompt']}\" ---")
        print(f"Scenario: {q['desc']}")
        
        results = engine.search(query=q["prompt"], limit=2)
        if not results:
            print("  [!] No documents matched.")
        else:
            for rank, r in enumerate(results, 1):
                display_search_result(rank, r)

    # 3. Interactive Search Mode
    print_banner("[3/3] INTERACTIVE REAL-TIME SEARCH (Type 'exit' to quit)")
    print("You can now test any custom query against your custom documents in Qdrant:")
    
    try:
        while True:
            user_query = input("\nEnter your search query (or 'exit'): ").strip()
            if not user_query or user_query.lower() in ["exit", "quit", "q"]:
                print("\nExiting search demo. Goodbye!")
                break
                
            results = engine.search(query=user_query, limit=3)
            if not results:
                print("  [!] No matching chunks found.")
            else:
                for rank, r in enumerate(results, 1):
                    display_search_result(rank, r)
    except (KeyboardInterrupt, EOFError):
        print("\nSession ended.")
    finally:
        qdrant_mgr.close()

if __name__ == "__main__":
    run_enterprise_demo()
