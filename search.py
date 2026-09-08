import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from core.search_engine import VectorSearchEngine
from core.qdrant_manager import QdrantManager

def main():
    print("=" * 80)
    print("  🔍 QDRANT CUSTOM SEMANTIC SEARCH ENGINE (3,072-DIMENSIONS)")
    print("=" * 80)
    print("Connected to your live Qdrant Cloud cluster.\n")
    
    qdrant_mgr = QdrantManager()
    engine = VectorSearchEngine(qdrant_mgr=qdrant_mgr)
    
    print("👉 Type any question you want to ask your documents (or 'exit' to quit):\n")
    
    while True:
        try:
            query = input("Ask a question > ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("\nExiting search. Goodbye!")
                break
                
            results = engine.search(query=query, limit=3)
            
            if not results:
                print("\n  [!] No matching passages found in Qdrant.\n")
                continue
                
            print(f"\nTop {len(results)} Relevant Results from Qdrant Cloud:\n" + "-" * 80)
            for i, r in enumerate(results, 1):
                page_info = f" (Page {r['page_number']})" if r.get('page_number') else ""
                print(f"[{i}] File: {r.get('file_name', 'Document')}{page_info}")
                print(f"    Similarity Score: {r['score']:.4f} (Cosine Similarity)")
                print(f"    Passage:\n    {r['text'].strip()}")
                print("-" * 80)
            print()
        except (KeyboardInterrupt, EOFError):
            print("\n\nSession ended.")
            break

    qdrant_mgr.close()

if __name__ == "__main__":
    main()
