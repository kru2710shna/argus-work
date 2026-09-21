"""Exact inner-product index in plain numpy, for machines without faiss.

Same contract and results as FaissFlatIndex (IndexFlatIP: exact cosine search
over L2-normalized descriptors), just slower. At Alps scale (211,804 vectors)
a 512-d search is ~100M multiply-adds per query, fine for an offline eval.
Saves to <path>.npy instead of <path>.faiss, so the two cache formats don't
mix; load() says which one it found.
"""

import json
import os

import numpy as np

from core.interfaces import DescriptorIndex


class NumpyFlatIndex(DescriptorIndex):
    def __init__(self, descriptor_dim: int):
        self.descriptor_dim = descriptor_dim
        self.tile_ids: list[str] = []
        self._chunks: list[np.ndarray] = []
        self._matrix: np.ndarray | None = None

    @property
    def ntotal(self) -> int:
        return len(self.tile_ids)

    def _vectors(self) -> np.ndarray:
        if self._matrix is None:
            self._matrix = (
                np.concatenate(self._chunks)
                if self._chunks
                else np.empty((0, self.descriptor_dim), dtype=np.float32)
            )
            self._chunks = [self._matrix]
        return self._matrix

    def add(self, tile_ids: list[str], descriptors: np.ndarray) -> None:
        descriptors = np.ascontiguousarray(descriptors, dtype=np.float32)
        if descriptors.shape[1] != self.descriptor_dim:
            raise ValueError(
                f"expected descriptors of dim {self.descriptor_dim}, got {descriptors.shape[1]}"
            )
        self._chunks.append(descriptors)
        self._matrix = None
        self.tile_ids.extend(tile_ids)

    def search(self, q: np.ndarray, k: int) -> list[tuple[str, float]]:
        vectors = self._vectors()
        k = min(k, len(vectors))
        if k == 0:
            return []
        similarities = vectors @ np.asarray(q, dtype=np.float32).reshape(-1)
        top = np.argpartition(-similarities, k - 1)[:k]
        # Ties broken by insertion order, like faiss's flat index.
        top = top[np.lexsort((top, -similarities[top]))]
        return [(self.tile_ids[i], float(similarities[i])) for i in top]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.save(f"{path}.npy", self._vectors())
        with open(f"{path}.tile_ids.json", "w") as f:
            json.dump(self.tile_ids, f)

    def load(self, path: str) -> None:
        if not os.path.exists(f"{path}.npy"):
            found = " (a faiss-format cache is there; install faiss or rebuild)" if os.path.exists(f"{path}.faiss") else ""
            raise FileNotFoundError(f"no {path}.npy{found}")
        self._matrix = np.load(f"{path}.npy")
        self._chunks = [self._matrix]
        self.descriptor_dim = self._matrix.shape[1]
        with open(f"{path}.tile_ids.json") as f:
            self.tile_ids = json.load(f)
