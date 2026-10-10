"""
app.py
------
FastAPI deployment layer for the RAG-Powered Search Engine.

Endpoints:
  GET  /health          - liveness check
  POST /ingest           - pull fresh papers from arXiv into the corpus
  POST /build-index       - (re)build the FAISS index from the corpus
  POST /query             - ask a question, get an answer + cited sources
  GET  /                  - minimal demo UI (static/index.html)

Run:
  uv run uvicorn app:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Load .env BEFORE importing anything that reads env vars at import/init time
# (e.g. rag.generator reads GOOGLE_API_KEY, app.py below reads EMBED_DEVICE).
load_dotenv()

from ingest import run as ingest_run
from rag.pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("app")

BASE_DIR = Path(__file__).parent
CORPUS_PATH = BASE_DIR / "data" / "corpus.jsonl"
INDEX_DIR = BASE_DIR / "data" / "index"

app = FastAPI(
    title="RAG-Powered Search Engine",
    description="Retrieval-Augmented Generation Q&A over arXiv paper abstracts.",
    version="1.0.0",
)

# EMBED_DEVICE lets you force a device ("cpu", "mps", "cuda") via env var if
# auto-detection picks the wrong one. Leave unset to auto-select (MPS on
# Apple Silicon, CUDA on NVIDIA, else CPU).
from rag.embeddings import Embedder

_embed_device = os.environ.get("EMBED_DEVICE") or None
pipeline = RAGPipeline(embedder=Embedder(device=_embed_device))


@app.on_event("startup")
def load_index_if_present():
    if (INDEX_DIR / "index.faiss").exists():
        logger.info("Loading existing FAISS index...")
        pipeline.load_index(str(INDEX_DIR))
        logger.info("Index loaded.")
    else:
        logger.warning("No index found yet. Call /ingest then /build-index before querying.")


class IngestRequest(BaseModel):
    query: str = Field(default="retrieval augmented generation", description="arXiv search query")
    max_results: int = Field(default=60, ge=1, le=300)


class QueryRequest(BaseModel):
    question: str
    k: int = Field(default=5, ge=1, le=20)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "index_loaded": pipeline.store is not None,
        "indexed_chunks": len(pipeline.store.metadata) if pipeline.store else 0,
        "gemini_configured": bool(os.environ.get("GOOGLE_API_KEY")),
        "embedding_device": pipeline.embedder.device,
    }


@app.post("/ingest")
def ingest(req: IngestRequest):
    records = ingest_run(req.query, req.max_results, str(CORPUS_PATH))
    return {"papers_ingested": len(records), "corpus_path": str(CORPUS_PATH)}


@app.post("/build-index")
def build_index():
    if not CORPUS_PATH.exists():
        raise HTTPException(400, "No corpus found. Call /ingest first.")
    n_chunks = pipeline.build_index_from_corpus(str(CORPUS_PATH), str(INDEX_DIR))
    return {"chunks_indexed": n_chunks, "index_dir": str(INDEX_DIR)}


@app.post("/query")
def query(req: QueryRequest):
    if pipeline.store is None:
        raise HTTPException(400, "Index not built yet. Call /ingest then /build-index first.")
    return pipeline.answer(req.question, k=req.k)


static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
def root():
    index_html = static_dir / "index.html"
    if index_html.exists():
        return FileResponse(str(index_html))
    return {"message": "RAG Search Engine API. See /docs for Swagger UI."}
