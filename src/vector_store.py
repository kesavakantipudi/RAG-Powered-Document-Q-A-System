import faiss
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
