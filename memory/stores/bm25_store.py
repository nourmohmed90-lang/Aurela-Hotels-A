from typing import List, Dict, Optional
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi

from memory.stores.config import VECTOR_DB_DIR


class BM25Store:

    def __init__(self):
        self.documents = []
        self.bm25 = None
        self.index_path = VECTOR_DB_DIR / "bm25_index.pkl"

    def build(self, documents: List[Dict]):
        if not documents:
            return

        self.documents = documents
        corpus = [self.tokenize(doc["text"]) for doc in documents]
        self.bm25 = BM25Okapi(corpus)

    def save(self):
        data = {
            "documents": self.documents,
            "bm25": self.bm25
        }
        with open(self.index_path, "wb") as f:
            pickle.dump(data, f)

    def load(self):
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found: {self.index_path}"
            )

        with open(self.index_path, "rb") as f:
            data = pickle.load(f)

        self.documents = data["documents"]
        self.bm25 = data["bm25"]

    def search(
        self,
        query: str,
        k: int = 5,
        source: Optional[str] = None
    ) -> List[Dict]:
        if self.bm25 is None:
            self.load()

        query_tokens = self.tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        results = []
        for doc, score in zip(self.documents, scores):
            if source is not None and doc["metadata"].get("source") != source:
                continue

            results.append({
                "text": doc["text"],
                "metadata": doc["metadata"],
                "score": float(score)
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    def count(self):
        return len(self.documents)

    @staticmethod
    def tokenize(text: str):
        return text.lower().split()

    def info(self):
        return {
            "documents": len(self.documents),
            "path": str(self.index_path)
        }


bm25_store = BM25Store()