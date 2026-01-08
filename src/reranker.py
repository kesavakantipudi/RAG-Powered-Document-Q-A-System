import os
from typing import List


class OptionalCrossEncoderReranker:
    """Optional reranker; uses CrossEncoder when enabled and available."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.enabled = os.getenv("RAG_USE_RERANKER", "0") not in ("0", "false", "False")
        self.model = None
        if self.enabled:
            try:
                from sentence_transformers import CrossEncoder  # type: ignore

                self.model = CrossEncoder(model_name)
            except Exception:
                self.enabled = False

    def rerank(self, query: str, chunks: List[dict], top_k: int) -> List[dict]:
        if not self.enabled or self.model is None or not chunks:
            return chunks[:top_k]

        pairs = [(query, c["chunk_text"]) for c in chunks]
        scores = self.model.predict(pairs)
        scored = list(zip(chunks, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[:top_k]]
