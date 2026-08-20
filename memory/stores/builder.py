import time
from pathlib import Path
from typing import Dict, Any

from memory.stores.chunker import load_documents, split_document
from memory.stores.embeddings import encode
from memory.stores.vector_store import vector_store
from memory.stores.bm25_store import bm25_store
from memory.stores.config import DOCUMENTS_DIR


def build_vector_database(reset: bool = True) -> Dict[str, Any]:
    start = time.time()

    print("=" * 60)
    print("Building Vector & Sparse Database")
    print("=" * 60)

    if reset:
        print("[1/6] Resetting ChromaDB collection...")
        vector_store.reset()

    print("[2/6] Loading documents...")
    documents = load_documents(DOCUMENTS_DIR)  # skips unreadable files, doesn't crash the batch

    if not documents:
        # Empty corpus is valid (e.g. the admin just deleted the last doc).
        # Both indexes must reflect that -- don't leave BM25 holding a stale
        # in-memory index from before the deletion.
        print(f"No documents found in target path: {DOCUMENTS_DIR}")
        bm25_store.build([])
        bm25_store.save()
        elapsed = round(time.time() - start, 2)
        print(f"Knowledge base is now empty. Time Taken: {elapsed}s")
        return {"indexed_chunks": 0, "vector_docs": vector_store.count(),
                "bm25_docs": bm25_store.count(), "elapsed_seconds": elapsed}

    print(f"Loaded {len(documents)} document chunks.")

    print("[3/6] Generating embeddings...")
    texts = [doc["text"] for doc in documents]
    embeddings = encode(texts)

    for doc, embedding in zip(documents, embeddings):
        doc["embedding"] = embedding.tolist()

    print("[4/6] Persisting vectors to ChromaDB...")
    vector_store.add_documents(documents)

    print("[5/6] Building BM25 sparse index...")
    bm25_store.build(documents)
    bm25_store.save()

    elapsed = round(time.time() - start, 2)
    print("=" * 60)
    print("Knowledge Base Constructed Successfully")
    print("=" * 60)
    print(f"Indexed Chunks : {len(documents)}")
    print(f"Vector Docs    : {vector_store.count()}")
    print(f"BM25 Docs      : {bm25_store.count()}")
    print(f"Time Taken     : {elapsed}s")
    print("=" * 60)

    return {"indexed_chunks": len(documents), "vector_docs": vector_store.count(),
            "bm25_docs": bm25_store.count(), "elapsed_seconds": elapsed}


def validate_document(path: Path) -> None:
    split_document(path)  # raises on malformed/corrupt content


def sync_documents(reset: bool = True) -> Dict[str, Any]:
    return build_vector_database(reset=reset)


if __name__ == "__main__":
    build_vector_database()
