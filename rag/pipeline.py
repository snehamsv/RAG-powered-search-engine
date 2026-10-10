"""
pipeline.py
-----------
Orchestrates the end-to-end RAG flow:
  corpus (JSONL) -> chunk -> embed -> FAISS index -> retrieve -> generate

This is the single object both the notebook and the FastAPI app import,
so behavior stays identical between exploration and the deployed app.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .chunking import chunk_text
from .embeddings import Embedder
from .generator import generate_answer
from .vectorstore import VectorStore

logger = logging.getLogger("rag.pipeline")


class RAGPipeline:
    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or Embedder()
        self.store: VectorStore | None = None

    # ---------- Indexing ----------

    def build_index_from_corpus(self, corpus_path: str, index_dir: str) -> int:
        records = self._load_corpus(corpus_path)
        chunk_meta = []
        chunk_texts = []

        for r in records:
            pieces = chunk_text(r["abstract"])
            for i, piece in enumerate(pieces):
                chunk_meta.append({
                    "paper_id": r["id"],
                    "title": r["title"],
                    "text": piece,
                    "chunk_idx": i,
                    "source": r.get("source", "unknown"),
                })
                chunk_texts.append(piece)

        logger.info(f"Embedding {len(chunk_texts)} chunks from {len(records)} papers")
        vectors = self.embedder.encode(chunk_texts)

        self.store = VectorStore(dim=self.embedder.dim)
        self.store.add(vectors, chunk_meta)
        self.store.save(index_dir)
        return len(chunk_texts)

    def load_index(self, index_dir: str) -> None:
        self.store = VectorStore.load(index_dir)

    # ---------- Querying ----------

    def retrieve(self, question: str, k: int = 5) -> list[dict]:
        if self.store is None:
            raise RuntimeError("Index not loaded. Call build_index_from_corpus() or load_index() first.")
        q_vec = self.embedder.encode([question])[0]
        return self.store.search(q_vec, k=k)

    def answer(self, question: str, k: int = 5) -> dict:
        chunks = self.retrieve(question, k=k)
        answer_text = generate_answer(question, chunks)
        return {
            "question": question,
            "answer": answer_text,
            "sources": [
                {"title": c["title"], "paper_id": c["paper_id"], "score": c["score"]}
                for c in chunks
            ],
        }

    @staticmethod
    def _load_corpus(corpus_path: str) -> list[dict]:
        records = []
        with open(corpus_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
