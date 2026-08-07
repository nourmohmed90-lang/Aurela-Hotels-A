from typing import List, Dict, Optional

from memory.stores.config import TOP_K
from memory.stores.embeddings import encode_query
from memory.stores.vector_store import vector_store
from memory.stores.bm25_store import bm25_store


class HybridRetriever:

    def __init__(self):
        self.vector_store = vector_store
        self.bm25_store = bm25_store

    @staticmethod
    def normalize(results: List[Dict]) -> List[Dict]:
        if not results:
            return results

        scores = [doc["score"] for doc in results]
        max_score, min_score = max(scores), min(scores)

        if max_score == min_score:
            for doc in results:
                doc["normalized_score"] = 1.0
            return results

        for doc in results:
            doc["normalized_score"] = (doc["score"] - min_score) / (max_score - min_score)

        return results

    def retrieve(
        self,
        query: str,
        k: int = TOP_K,
        source: Optional[str] = None,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5
    ) -> List[Dict]:
        query_embedding = encode_query(query)

        vector_results = self.vector_store.search(
            embedding=query_embedding,
            k=k * 2,
            source=source
        )

        bm25_results = self.bm25_store.search(
            query=query,
            k=k * 2,
            source=source
        )

        vector_results = self.normalize(vector_results)
        bm25_results = self.normalize(bm25_results)

        merged = {}

        for doc in vector_results:
            key = (doc["metadata"]["source"], doc["metadata"]["chunk"])
            merged[key] = {
                "text": doc["text"],
                "metadata": doc["metadata"],
                "vector_score": doc["normalized_score"],
                "bm25_score": 0.0
            }

        for doc in bm25_results:
            key = (doc["metadata"]["source"], doc["metadata"]["chunk"])
            if key not in merged:
                merged[key] = {
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "vector_score": 0.0,
                    "bm25_score": doc["normalized_score"]
                }
            else:
                merged[key]["bm25_score"] = doc["normalized_score"]

        results = []
        for item in merged.values():
            item["score"] = (
                vector_weight * item["vector_score"] +
                bm25_weight * item["bm25_score"]
            )
            results.append(item)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    def retrieve_context(
        self,
        query: str,
        k: int = TOP_K,
        source: Optional[str] = None
    ) -> str:
        documents = self.retrieve(query=query, k=k, source=source)
        if not documents:
            return ""

        context = []
        for doc in documents:
            context.append(
                f"SOURCE: {doc['metadata']['source']}\n"
                f"CHUNK: {doc['metadata']['chunk']}\n"
                f"SCORE: {doc['score']:.3f}\n"
                f"CONTENT: {doc['text']}\n"
            )

        return "\n".join(context)


hybrid_retriever = HybridRetriever()