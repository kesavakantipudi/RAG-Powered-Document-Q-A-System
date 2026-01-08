import os
import numpy as np


class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._use_st = os.getenv("RAG_USE_SENTENCE_TRANSFORMERS", "1") not in ("0", "false", "False")
        self._st_model = None
        self._dim = 384  # default dim for fallback

        if self._use_st:
            try:
                from sentence_transformers import SentenceTransformer  # local import to avoid hard dependency
                self._st_model = SentenceTransformer(model_name)
            except Exception:
                self._use_st = False

    def _fallback_embed(self, texts):
        # Deterministic hashing-based embedding: simple, dependency-free, CPU-fast.
        # Not semantically meaningful, but sufficient for tests/pipeline wiring.
        dim = self._dim
        vectors = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            if not t:
                continue
            acc = np.zeros(dim, dtype=np.float32)
            # Use a basic bag-of-words style accumulation with hashing to dims
            for token in t.split():
                h = hash(token)  # deterministic within a single process run
                idx = h % dim
                val = ((h >> 8) & 0xFFFF) / 65535.0  # pseudo value in [0,1]
                acc[idx] += 1.0 + val
            # Normalize to unit length to behave like real embeddings
            norm = np.linalg.norm(acc)
            vectors[i] = acc / norm if norm > 0 else acc
        return vectors

    def embed_texts(self, texts):
        if self._use_st and self._st_model is not None:
            try:
                emb = self._st_model.encode(texts, show_progress_bar=False)
                # Ensure float32 for faiss
                return np.asarray(emb, dtype=np.float32)
            except Exception:
                return self._fallback_embed(texts)
        return self._fallback_embed(texts)
