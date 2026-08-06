from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings

from .config import (
    VECTOR_DB_DIR,
    COLLECTION_NAME,
)


class VectorStore:

    def __init__(self):

        self.client = chromadb.PersistentClient(
            path=str(VECTOR_DB_DIR),
            settings=Settings(
                anonymized_telemetry=False
            )
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "description": "Aurelia Hotels Knowledge Base",
                "hnsw:space": "cosine"
            }
        )

    # ---------------------------------------------------
    # Add Documents
    # ---------------------------------------------------

    def add_documents(self, documents: List[Dict]):

        if not documents:
            return

        ids = []
        texts = []
        embeddings = []
        metadatas = []

        for doc in documents:

            ids.append(doc["id"])

            texts.append(doc["text"])

            embeddings.append(doc["embedding"])

            metadatas.append(doc["metadata"])

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

    # ---------------------------------------------------
    # Search
    # ---------------------------------------------------

    def search(
        self,
        embedding,
        k: int = 3,
        source: Optional[str] = None
    ):

        kwargs = {
            "query_embeddings": [embedding.tolist()],
            "n_results": k
        }

        if source is not None:

            kwargs["where"] = {
                "source": source
            }

        results = self.collection.query(**kwargs)

        output = []

        docs = results["documents"][0]
        meta = results["metadatas"][0]
        dist = results["distances"][0]

        for d, m, s in zip(docs, meta, dist):

            output.append(
                {
                    "text": d,
                    "metadata": m,
                    "distance": float(s),
                    "score": float(1 - s)
                }
            )

        return output

    # ---------------------------------------------------
    # Count
    # ---------------------------------------------------

    def count(self):

        return self.collection.count()

    # ---------------------------------------------------
    # Delete Collection
    # ---------------------------------------------------

    def reset(self):

        self.client.delete_collection(COLLECTION_NAME)

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "description": "Aurelia Hotels Knowledge Base",
                "hnsw:space": "cosine"
            }
        )

    # ---------------------------------------------------
    # Collection Info
    # ---------------------------------------------------

    def info(self):

        return {
            "collection": COLLECTION_NAME,
            "documents": self.count(),
            "path": str(VECTOR_DB_DIR)
        }


vector_store = VectorStore()