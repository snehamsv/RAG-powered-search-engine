"""
embeddings.py
-------------
Wraps a local sentence-transformers model so the rest of the pipeline
doesn't care which embedding backend is used.

Auto-selects the fastest available device:
  - Apple Silicon (M1/M2/M3) -> "mps" (Metal GPU backend)
  - NVIDIA GPU               -> "cuda"
  - otherwise                -> "cpu"
On an 8GB M1 Mac, MPS gives a large speedup over CPU for embedding
batches while using the on-chip GPU instead of competing with the CPU
for the rest of the pipeline (FastAPI, FAISS search, etc).

Swap-in note for GCP production: replace this class with a call to
Vertex AI's `text-embedding-004` model (via `google-cloud-aiplatform`)
and keep the same `.encode()` interface - nothing else in the RAG
pipeline needs to change.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("rag.embeddings")

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Kept small deliberately: on an 8GB unified-memory Mac, large batches
# competing with the OS + FastAPI + FAISS for RAM can cause swapping or
# MPS OOM. 16-32 is a safe, still-fast range for this model size.
DEFAULT_BATCH_SIZE = 16


def resolve_device(preferred: str | None = None) -> str:
    """Pick the best available torch device. Pass preferred='cpu' to force CPU."""
    if preferred:
        return preferred
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class Embedder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        # Imported lazily so this module can be imported (e.g. for testing
        # other pipeline pieces) without pulling in the heavy torch
        # dependency unless an Embedder is actually instantiated.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = resolve_device(device)
        self.batch_size = batch_size
        logger.info(f"Loading embedding model '{model_name}' on device='{self.device}'")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=self.batch_size,
        )
        return vecs.astype("float32")
