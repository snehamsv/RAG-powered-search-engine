"""
generator.py
------------
Answer synthesis layer. Uses Google's Gemini API when GOOGLE_API_KEY is
set (ties this project to the GCP/Vertex ecosystem - Gemini is the
same model family served via Vertex AI in production). If no key is
configured, falls back to a simple extractive summary so the whole
app still runs end-to-end with zero paid setup.
"""

from __future__ import annotations

import concurrent.futures
import os

GEMINI_TIMEOUT_SECONDS = 20

SYSTEM_PROMPT = (
    "You are a research assistant. Answer the user's question ONLY using "
    "the provided context snippets from research paper abstracts. "
    "Cite paper titles inline when you use a fact. If the context does not "
    "contain the answer, say so plainly instead of guessing."
)


def _build_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        parts.append(f"[{c.get('title', 'Untitled')}]: {c.get('text', c.get('abstract', ''))}")
    return "\n\n".join(parts)


def _extractive_fallback(question: str, chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant context was retrieved for this question."
    lines = [
        "(Running in extractive fallback mode - set GOOGLE_API_KEY for LLM-generated answers.)",
        "",
        f"Top matches for: \"{question}\"",
    ]
    for c in chunks:
        snippet = c.get("text", c.get("abstract", ""))[:280]
        lines.append(f"- {c.get('title', 'Untitled')} (score={c.get('score', 0):.3f}): {snippet}...")
    return "\n".join(lines)


def _call_gemini(api_key: str, prompt: str) -> str:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    #model = genai.GenerativeModel("gemini-1.5-flash")
    # Working line
    model = genai.GenerativeModel("gemini-3.6-flash")
    # request_options timeout is a belt; the ThreadPoolExecutor wrapper in
    # generate_answer() below is the suspenders, in case the SDK doesn't
    # honor it on every transport path.
    resp = model.generate_content(
        prompt,
        request_options={"timeout": GEMINI_TIMEOUT_SECONDS},
    )
    return resp.text.strip()


_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="gemini-call")


def generate_answer(question: str, chunks: list[dict]) -> str:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return _extractive_fallback(question, chunks)

    context = _build_context(chunks)
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    # Submitted to a shared, long-lived executor (NOT a `with`-block executor):
    # a `with ThreadPoolExecutor()` blocks on __exit__ until the submitted
    # thread finishes, which defeats future.result(timeout=...) entirely if
    # the call hangs. A shared pool lets us abandon a stuck future and move on;
    # the orphaned thread is daemonized away when the process exits.
    future = _executor.submit(_call_gemini, api_key, prompt)
    try:
        return future.result(timeout=GEMINI_TIMEOUT_SECONDS + 5)
    except concurrent.futures.TimeoutError:
        fallback = _extractive_fallback(question, chunks)
        return f"(Gemini call timed out after {GEMINI_TIMEOUT_SECONDS}s. Falling back to extractive mode.)\n\n{fallback}"
    except Exception as e:
        fallback = _extractive_fallback(question, chunks)
        return f"(Gemini call failed: {e}. Falling back to extractive mode.)\n\n{fallback}"
