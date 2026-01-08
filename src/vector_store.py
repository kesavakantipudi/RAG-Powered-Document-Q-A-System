import faiss
import json
import os
import numpy as np


class VectorStore:
    def __init__(self, embedding_dim):
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.metadata = []

    def add(self, embeddings, metadata):
        self.index.add(np.array(embeddings).astype("float32"))
        self.metadata.extend(metadata)

    def search(self, query_embedding, top_k=5):
        distances, indices = self.index.search(
            np.array([query_embedding]).astype("float32"),
            top_k
        )

        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.metadata):
                results.append(self.metadata[idx])

        return results

    def save(self, index_path: str):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        meta_path = f"{index_path}.meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)

    @classmethod
    def load(cls, index_path: str):
        if not os.path.exists(index_path):
            return None
        meta_path = f"{index_path}.meta.json"
        if not os.path.exists(meta_path):
            return None

        index = faiss.read_index(index_path)
        store = cls(embedding_dim=index.d)
        store.index = index
        with open(meta_path, "r", encoding="utf-8") as f:
            store.metadata = json.load(f)
        return store
