import time

from .chunker import load_documents
from .embeddings import encode
from .vector_store import vector_store
from .bm25_store import bm25_store
from .config import DOCUMENTS_DIR


def build_vector_database(reset: bool = True):

    start = time.time()

    print("=" * 60)
    print("Building Aurelia Hotels Vector Database")
    print("=" * 60)
# Reset ChromaDB
    if reset:
        print("[1/6] Resetting collection...")
        vector_store.reset()
# Load Documents
    print("[2/6] Loading documents...")
    documents = load_documents(DOCUMENTS_DIR)

    if not documents:
        print("No documents found.")
        return

    print(f"Loaded {len(documents)} chunks.")
# Create Embeddings
    print("[3/6] Creating embeddings...")

    texts = [doc["text"] for doc in documents]

    embeddings = encode(texts)

    for doc, embedding in zip(documents, embeddings):

        doc["embedding"] = embedding.tolist()
# Save into ChromaDB
    print("[4/6] Saving into ChromaDB...")

    vector_store.add_documents(documents)

# Build BM25 Index
    print("[5/6] Building BM25 index...")

    bm25_store.build(documents)
    bm25_store.save()

    # Finished
    print("[6/6] Finished")
    elapsed = round(time.time() - start, 2)

    print()
    print("=" * 60)
    print("Knowledge Base Created Successfully")
    print("=" * 60)
    print(f"Indexed Chunks : {len(documents)}")
    print(f"Vector Docs    : {vector_store.count()}")
    print(f"BM25 Docs      : {bm25_store.count()}")
    print(f"Collection     : {vector_store.info()['collection']}")
    print(f"Time           : {elapsed} sec")
    print("=" * 60)


if __name__ == "__main__":
    build_vector_database()
