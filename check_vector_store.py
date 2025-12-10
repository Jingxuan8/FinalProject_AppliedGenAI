import chromadb
import os

VECTOR_PATH = "data/vector_store"

def main():
    print("=== CHECKING VECTOR STORE ===")
    print("Path:", os.path.abspath(VECTOR_PATH))

    client = chromadb.PersistentClient(path=VECTOR_PATH)

    collections = client.list_collections()
    print("\nCollections:", collections)

    if not collections:
        print("\n❌ No collections found — vector store may be empty or misconfigured.")
        return

    col = collections[0]
    print(f"\nUsing collection: {col.name}")

    # Fetch 5 items to inspect
    items = col.get(limit=5, include=["documents", "embeddings", "metadatas"])
    print("\n=== SAMPLE DOCUMENTS ===")
    print(items)

    # Test search
    res = col.query(query_texts=["board game"], n_results=5)
    print("\n=== QUERY RESULTS ===")
    print(res)

if __name__ == "__main__":
    main()
