from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL


class EmbeddingModel:
    """
    Singleton wrapper around SentenceTransformer.

    The model is loaded only once during the application lifetime.
    """

    _model = None

    @classmethod
    def model(cls):

        if cls._model is None:

            print(f"[Embedding] Loading model: {EMBEDDING_MODEL}")

            cls._model = SentenceTransformer(
                EMBEDDING_MODEL
            )

        return cls._model


def encode(texts: List[str]) -> np.ndarray:
    """
    Encode a list of texts into dense vectors.
    """

    model = EmbeddingModel.model()

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings


def encode_query(query: str) -> np.ndarray:
    """
    Encode a single user query.
    """

    return encode([query])[0]