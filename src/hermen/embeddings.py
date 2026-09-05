from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from hermen.config import EmbeddingConfig


class Embedder(Protocol):
    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        ...

    def embed_query(self, text: str) -> np.ndarray:
        ...


class HashEmbedder:
    def __init__(self, dimensions: int = 384) -> None:
        if dimensions < 1:
            raise ValueError("Embedding dimensions must be positive")
        self.dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(text)

    def _embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = np.linalg.norm(vector)
        if norm != 0:
            vector /= norm
        return vector


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Install with: uv pip install -e '.[local]'"
            ) from exc

        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return [np.asarray(vector, dtype=np.float32) for vector in vectors]

    def embed_query(self, text: str) -> np.ndarray:
        vector = self.model.encode([text], normalize_embeddings=True)[0]
        return np.asarray(vector, dtype=np.float32)


def build_embedder(config: EmbeddingConfig) -> Embedder:
    if config.provider == "hash":
        return HashEmbedder(dimensions=config.dimensions)
    if config.provider == "sentence_transformers":
        return SentenceTransformerEmbedder(config.model)
    raise ValueError(f"Unsupported embedding provider: {config.provider}")
