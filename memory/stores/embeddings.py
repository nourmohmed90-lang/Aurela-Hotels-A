from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

from memory.stores.config import EMBEDDING_MODEL


class EmbeddingModel:
    _model = None

    @classmethod
    def model(cls):
        if cls._model is None:
            print(f"[Embedding] Loading model: {EMBEDDING_MODEL}")
            cls._model = SentenceTransformer(EMBEDDING_MODEL)
        return cls._model


def encode(texts: List[str]) -> np.ndarray:
    model = EmbeddingModel.model()
    return model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )


def encode_query(query: str) -> np.ndarray:
    return encode([query])[0]