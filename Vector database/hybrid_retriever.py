from typing import List, Dict, Optional

from .config import TOP_K
from .embeddings import encode_query
from .vector_store import vector_store
from .bm25_store import bm25_store


class HybridRetriever:

    def __init__(self):

        self.vector_store = vector_store
        self.bm25_store = bm25_store

    # Normalize Scores

    @staticmethod
    def normalize(results: List[Dict]) -> List[Dict]:

        if not results:
            return results

        scores = [doc["score"] for doc in results]

        max_score = max(scores)
        min_score = min(scores)

        if max_score == min_score:

            for doc in results:
                doc["normalized_score"] = 1.0

            return results

        for doc in results:

            doc["normalized_score"] = (
                (doc["score"] - min_score)
                / (max_score - min_score)
            )

        return results

    # Hybrid Search

    def retrieve(
        self,
        query: str,
        k: int = TOP_K,
        source: Optional[str] = None,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5
    ) -> List[Dict]:

        # Vector Search

        query_embedding = encode_query(query)

        vector_results = self.vector_store.search(
            embedding=query_embedding,
            k=k * 2,
            source=source
        )

        # BM25 Search

        bm25_results = self.bm25_store.search(
            query=query,
            k=k * 2,
            source=source
        )

        # Normalize Scores

        vector_results = self.normalize(vector_results)
        bm25_results = self.normalize(bm25_results)

        # Merge Results

        merged = {}

        # Vector documents

        for doc in vector_results:

            key = (
                doc["metadata"]["source"],
                doc["metadata"]["chunk"]
            )

            merged[key] = {
                "text": doc["text"],
                "metadata": doc["metadata"],
                "vector_score": doc["normalized_score"],
                "bm25_score": 0.0
            }

        # BM25 documents

        for doc in bm25_results:

            key = (
                doc["metadata"]["source"],
                doc["metadata"]["chunk"]
            )

            if key not in merged:

                merged[key] = {
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "vector_score": 0.0,
                    "bm25_score": doc["normalized_score"]
                }

            else:

                merged[key]["bm25_score"] = doc["normalized_score"]

        # Final Score

        results = []

        for item in merged.values():

            item["score"] = (
                vector_weight * item["vector_score"]
                +
                bm25_weight * item["bm25_score"]
            )

            results.append(item)

        # Sort

        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return results[:k]

    # Context Builder

    def retrieve_context(
        self,
        query: str,
        k: int = TOP_K,
        source: Optional[str] = None
    ) -> str:

        documents = self.retrieve(
            query=query,
            k=k,
            source=source
        )

        if not documents:
            return ""

        context = []

        for doc in documents:

            context.append(
                f"""
SOURCE:
{doc["metadata"]["source"]}

CHUNK:
{doc["metadata"]["chunk"]}

SCORE:
{doc["score"]:.3f}

CONTENT:
{doc["text"]}
"""
            )

        return "\n".join(context)


hybrid_retriever = HybridRetriever()
