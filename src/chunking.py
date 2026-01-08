"""Chunking utilities (char-based fallback and token-aware splitting)."""

from typing import List, Optional


def _chunk_by_chars(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = max(end - overlap, start + 1)  # ensure progress

    return chunks


def _chunk_by_tokens(text: str, tokenizer, chunk_size: int = 256, overlap: int = 32) -> List[str]:
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    n = len(tokens)

    while start < n:
        end = min(start + chunk_size, n)
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        # ensure progress even if overlap is large
        start = max(end - overlap, start + 1)

    return chunks


def chunk_text(text: str,
               chunk_size: int = 500,
               overlap: int = 100,
               tokenizer=None,
               chunk_size_tokens: int = 256,
               overlap_tokens: int = 32) -> List[str]:
    """Chunk text either by tokens (if tokenizer provided) or by chars."""
    if tokenizer is not None:
        try:
            return _chunk_by_tokens(text, tokenizer, chunk_size_tokens, overlap_tokens)
        except Exception:
            # Fall back silently to char-based chunking
            pass
    return _chunk_by_chars(text, chunk_size, overlap)


def chunk_documents(documents,
                    chunk_size: int = 500,
                    overlap: int = 100,
                    tokenizer=None,
                    chunk_size_tokens: int = 256,
                    overlap_tokens: int = 32):
    all_chunks = []

    for doc in documents:
        text_chunks = chunk_text(
            doc["text"],
            chunk_size=chunk_size,
            overlap=overlap,
            tokenizer=tokenizer,
            chunk_size_tokens=chunk_size_tokens,
            overlap_tokens=overlap_tokens,
        )

        for idx, chunk in enumerate(text_chunks):
            all_chunks.append({
                "chunk_text": chunk,
                "source": doc["source"],
                "page": doc["page"],
                "chunk_id": f"{doc['source']}_p{doc['page']}_c{idx}"
            })

    return all_chunks
