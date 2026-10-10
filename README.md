# RAG-Powered Search Engine — Q&A over Research Papers

Capstone Project (Option 2: RAG-Powered Search Engine — Q&A with Retrieval-Augmented Generation)

A Retrieval-Augmented Generation system that ingests research paper abstracts live from
the **arXiv API**, indexes them with local embeddings + FAISS, and answers natural-language
questions with an LLM-generated, source-cited response (Gemini, with an automatic extractive
fallback if no API key is set — the app always runs end-to-end).

Built to mirror a real GCP data-engineering pipeline: **ingest → transform/chunk → embed →
index → serve**, with each local component documented against its managed GCP equivalent
(see [Architecture](#architecture) below).

---

## Project structure

```
.
├── app.py                 # FastAPI deployment app
├── ingest.py               # arXiv API ingestion (Cloud-Function-style)
├── project.ipynb            # End-to-end notebook: problem statement → results
├── rag/
│   ├── chunking.py           # Text chunking with overlap
│   ├── embeddings.py          # Local sentence-transformers wrapper
│   ├── vectorstore.py          # FAISS index wrapper (save/load)
│   ├── generator.py             # Gemini answer synthesis + extractive fallback
│   └── pipeline.py               # Orchestrates ingest→chunk→embed→index→retrieve→generate
├── static/index.html          # Minimal demo UI served at /
├── data/                        # Created at runtime (corpus.jsonl, FAISS index)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Running on an Apple Silicon Mac (M1/M2/M3), 8GB RAM

This project is tuned to work comfortably on an 8GB M1 Mac:

- **GPU acceleration is automatic.** `rag/embeddings.py` auto-detects Apple's **MPS**
  (Metal) backend and runs the embedding model on the M1's GPU instead of the CPU —
  no config needed. You'll see `Embedding device: mps` logged on startup. If you ever
  need to force CPU (e.g. to debug an MPS-specific error), set `EMBED_DEVICE=cpu` in
  `.env`.
- **Batch size is capped at 16** (`DEFAULT_BATCH_SIZE` in `rag/embeddings.py`) so
  embedding a batch of chunks doesn't spike unified memory and cause swapping on an
  8GB machine. You can raise it (e.g. to 32) once you confirm headroom in Activity
  Monitor, but 16 is a safe default.
- **FAISS runs on CPU** (`faiss-cpu`) — there's no meaningful GPU FAISS path on Apple
  Silicon, and CPU FAISS is plenty fast at this corpus size (tens of thousands of
  chunks). It won't compete with MPS for the GPU.
- **`torch`'s first install is the heavy part** (~2-3GB download, MPS-enabled build is
  the default on macOS arm64 via pip/uv — no special flag needed). Make sure you have
  a few GB free before running `uv pip install -r requirements.txt`.
- If you notice memory pressure while running the notebook *and* the API server at
  the same time, close one before running the other — both load their own copy of the
  embedding model into memory.

### VS Code setup

The zip includes a `.vscode/` folder:
- `launch.json` — two ready-to-use debug configs: **"FastAPI: uvicorn (debug)"** and
  **"Ingest: arXiv corpus"**. Just set your interpreter (below) and hit F5.
- `settings.json` — points VS Code at the `uv`-managed `.venv` automatically.
- `extensions.json` — recommends the Python, Jupyter, and debugpy extensions (VS Code
  will prompt you to install them on open).

After running `uv venv` (step 2 below), select the interpreter in VS Code:
`Cmd+Shift+P` → **Python: Select Interpreter** → choose `./.venv/bin/python`.
Then `project.ipynb` will run using the same venv, and F5 on either launch config
will hit the right Python with the right packages.

---

## Quickstart with `uv`

[`uv`](https://docs.astral.sh/uv/) is a fast Python package/venv manager. If you don't have it yet:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# or via pip, if you already have Python
pip install uv
```

### 1. Unzip and enter the project

```bash
unzip Final_Capstone_Project.zip -d capstone
cd capstone
```

### 2. Create the virtual environment and install dependencies

Your Mac's default Python is 3.13, but `faiss-cpu` (and a few other ML packages)
don't yet ship 3.13 wheels — so pin the venv to **Python 3.12**. `uv` will download
and manage that Python version automatically; you don't need to touch your system
Python or conda setup.

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

> If you see an error like `No solution found ... faiss-cpu ... cp313`, it means the
> venv was created with 3.13 instead of 3.12 — delete it (`rm -rf .venv`) and re-run
> the `uv venv --python 3.12` command above.

> First run of `uv pip install` downloads `sentence-transformers` + `torch`, which is a
> few GB — this only happens once. `sentence-transformers` will also download the
> `all-MiniLM-L6-v2` model weights (~90MB) from Hugging Face the first time you build
> the index; you need normal internet access (huggingface.co, export.arxiv.org) for
> these two one-time downloads.

### 3. (Optional) Set up your Gemini API key for LLM-generated answers

```bash
cp .env.example .env
# then edit .env and paste in a free key from https://aistudio.google.com/app/apikey
```

Without a key, the app still works fully — it just returns extractive top-matches
instead of an LLM-written answer, clearly labeled as such in the response.

### 4. Ingest data and build the index

```bash
uv run python ingest.py --query "retrieval augmented generation" --max-results 60
```

This writes `data/corpus.jsonl`. Then build the FAISS index from it — either via the
notebook (Section 5) or a one-liner:

```bash
uv run python -c "from rag.pipeline import RAGPipeline; from rag.embeddings import Embedder; p=RAGPipeline(embedder=Embedder()); p.build_index_from_corpus('data/corpus.jsonl','data/index')"
```

### 5. Run the notebook (optional — this is the "project.ipynb" deliverable)

```bash
uv run jupyter notebook project.ipynb
```

Runs the ingestion → chunking → embedding → indexing → retrieval → generation →
evaluation flow end-to-end with explanations at each step.

### 6. Run the API server

```bash
uv run uvicorn app:app --reload --port 8000
```

Then open:
- **Demo UI:** http://localhost:8000/
- **Swagger docs:** http://localhost:8000/docs

### 7. Try it

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are common chunking strategies for RAG?", "k": 5}'
```

Or refresh the corpus with a different topic straight from the API:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"query": "large language model agents", "max_results": 40}'

curl -X POST http://localhost:8000/build-index
```

---

## Architecture

```
                     ┌─────────────────┐
   arXiv API ───────▶│   ingest.py     │──▶ data/corpus.jsonl
  (public, no key)    │ fetch/validate/  │
                      │   normalize      │
                      └─────────────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │  rag/chunking.py │  word-count chunks + overlap
                     └─────────────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │rag/embeddings.py │  sentence-transformers (local)
                     └─────────────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │rag/vectorstore.py│  FAISS flat index (cosine sim)
                     └─────────────────┘
                              │
                    query ──▶ retrieve top-k chunks
                              │
                              ▼
                     ┌─────────────────┐
                     │ rag/generator.py │  Gemini (or extractive fallback)
                     └─────────────────┘
                              │
                              ▼
                    FastAPI (app.py) ──▶ JSON answer + cited sources
                              │
                    static/index.html ── minimal browser demo
```

### Local → GCP production swap-in map

| Local component (this repo) | GCP-managed equivalent |
|---|---|
| `ingest.py` local script | Cloud Function / Cloud Run Job, landing data in GCS or BigQuery |
| `rag/embeddings.py` (sentence-transformers) | Vertex AI `text-embedding-004` |
| `rag/vectorstore.py` (FAISS flat index) | Vertex AI Vector Search (managed ANN index) |
| `rag/generator.py` (Gemini API key) | Gemini via Vertex AI (service-account auth, no raw API key) |
| `uvicorn app:app` local server | Cloud Run (containerized, autoscaling) |
| `data/corpus.jsonl` local file | BigQuery table or GCS bucket |

The interfaces (`.encode()`, `.add()`/`.search()`, `generate_answer()`) are written so
each swap is a drop-in replacement — no changes needed elsewhere in the pipeline.

---

## Testing & evaluation

`project.ipynb` Section 8 runs a small evaluation harness: retrieval relevance
(top-score threshold hit-rate) and per-query latency across a fixed set of test
questions, with a bar chart of latency saved to `retrieval_latency.png`.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No solution found ... faiss-cpu ... cp313` | Your venv used Python 3.13. Run `rm -rf .venv && uv venv --python 3.12 && uv pip install -r requirements.txt`. |
| `ModuleNotFoundError: sentence_transformers` | Run `uv pip install -r requirements.txt` again — the venv wasn't activated/used. |
| Index not found / `/query` returns 400 | Run the ingest + build-index steps (Section 4) before querying. |
| Gemini call fails / falls back unexpectedly | Check `.env` has a valid `GOOGLE_API_KEY` and it's loaded (`uv run` picks up `.env` automatically via `python-dotenv` inside the app). |
| arXiv fetch returns 0 results | arXiv rate-limits aggressive polling — wait a few seconds and retry, or lower `--max-results`. |
