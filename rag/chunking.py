"""
chunking.py
-----------
Simple word-count based chunker with overlap. Abstracts are short so
chunking mostly matters for the notebook demo where longer text
(e.g. concatenated multi-paper context) is used.
"""

from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 220, overlap: int = 40) -> list[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
        if start <= 0:
            break
    return chunks
