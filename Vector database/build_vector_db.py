import time

from .chunker import load_documents
from .embeddings import encode
from .vector_store import vector_store
from .config import DOCUMENTS_DIR


def build_vector_database(reset: bool = True):

    start = time.time()

    print("=" * 60)
    print("Building Aurelia Hotels Vector Database")
    print("=" * 60)

    if reset:
        print("[1/5] Resetting collection...")
        vector_store.reset()

    print("[2/5] Loading documents...")
    documents = load_documents(DOCUMENTS_DIR)

    if not documents:
        print("No documents found.")
        return

    print(f"Loaded {len(documents)} chunks.")

    print("[3/5] Creating embeddings...")

    texts = [doc["text"] for doc in documents]

    embeddings = encode(texts)

    for doc, embedding in zip(documents, embeddings):

        doc["embedding"] = embedding.tolist()

    print("[4/5] Saving into ChromaDB...")

    vector_store.add_documents(documents)

    print("[5/5] Finished")

    elapsed = round(time.time() - start, 2)

    print()
    print("=" * 60)
    print("Vector Database Created Successfully")
    print("=" * 60)

    print(f"Indexed Chunks : {len(documents)}")
    print(f"Collection     : {vector_store.info()['collection']}")
    print(f"Stored Docs    : {vector_store.count()}")
    print(f"Time           : {elapsed} sec")
    print("=" * 60)


if __name__ == "__main__":

    build_vector_database()