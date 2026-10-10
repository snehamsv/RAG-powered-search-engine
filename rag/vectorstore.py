"""
vectorstore.py
--------------
Thin wrapper around a FAISS flat index (cosine similarity via
normalized inner product), with save/load and a parallel metadata store.

Swap-in note for GCP production: replace with Vertex AI Vector Search
(a managed ANN index) - the `.add()` / `.search()` interface below
would stay the same, only the backend changes.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # inner product on normalized vecs = cosine sim
        self.metadata: list[dict] = []

    def add(self, vectors: np.ndarray, metadata: list[dict]) -> None:
        assert vectors.shape[0] == len(metadata)
        self.index.add(vectors)
        self.metadata.extend(metadata)

    def search(self, query_vec: np.ndarray, k: int = 5) -> list[dict]:
        scores, idxs = self.index.search(query_vec.reshape(1, -1), k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            item = dict(self.metadata[idx])
            item["score"] = float(score)
            results.append(item)
        return results

    def save(self, dir_path: str) -> None:
        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p / "index.faiss"))
        with open(p / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)
        with open(p / "config.json", "w") as f:
            json.dump({"dim": self.dim, "count": len(self.metadata)}, f)

    @classmethod
    def load(cls, dir_path: str) -> "VectorStore":
        p = Path(dir_path)
        with open(p / "config.json") as f:
            cfg = json.load(f)
        store = cls(cfg["dim"])
        store.index = faiss.read_index(str(p / "index.faiss"))
        with open(p / "metadata.pkl", "rb") as f:
            store.metadata = pickle.load(f)
        return store
