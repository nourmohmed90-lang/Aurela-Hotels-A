from typing import List, Dict, Optional

from .config import TOP_K
from .embeddings import encode_query
from .vector_store import vector_store


class Retriever:

    def __init__(self):

        self.store = vector_store

    def retrieve(
        self,
        query: str,
        k: int = TOP_K,
        source: Optional[str] = None
    ) -> List[Dict]:

        query_embedding = encode_query(query)

        return self.store.search(
            embedding=query_embedding,
            k=k,
            source=source
        )

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

CONTENT:
{doc["text"]}
"""
            )

        return "\n".join(context)


retriever = Retriever()