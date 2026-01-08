import os

from transformers import AutoTokenizer

from src.loaders import load_documents
from src.chunking import chunk_documents
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.prompt import build_prompt
from src.generator import LocalGenerator
from src.reranker import OptionalCrossEncoderReranker

class RAGPipeline:
    def __init__(
        self,
        docs_path,
        index_path: str = "data/index/faiss.index",
        chunk_size_chars: int = 500,
        chunk_overlap_chars: int = 100,
        chunk_size_tokens: int = 256,
        chunk_overlap_tokens: int = 32,
    ):
        self.generator = LocalGenerator()
        self.embedder = EmbeddingModel()
        self.reranker = OptionalCrossEncoderReranker()

        # Tokenizer for token-aware chunking (optional)
        self.tokenizer = None
        if os.getenv("RAG_USE_TOKENIZER", "1") not in ("0", "false", "False"):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            except Exception:
                self.tokenizer = None

        # Try loading persisted index first
        store = VectorStore.load(index_path)
        if store is not None:
            self.store = store
            self.metadata = store.metadata
            self.texts = [c.get("chunk_text", "") for c in self.metadata]
            return

        # Build index from scratch
        docs = load_documents(docs_path)
        chunks = chunk_documents(
            docs,
            chunk_size=chunk_size_chars,
            overlap=chunk_overlap_chars,
            tokenizer=self.tokenizer,
            chunk_size_tokens=chunk_size_tokens,
            overlap_tokens=chunk_overlap_tokens,
        )

        self.texts = [c["chunk_text"] for c in chunks]
        self.metadata = chunks

        embeddings = self.embedder.embed_texts(self.texts)

        self.store = VectorStore(embedding_dim=embeddings.shape[1])
        self.store.add(embeddings, self.metadata)

        # Persist to disk for reuse
        try:
            self.store.save(index_path)
        except Exception:
            pass

    def answer(self, question, top_k=3):
        query_embedding = self.embedder.embed_texts([question])[0]

        # Retrieve a slightly larger pool for potential reranking
        initial_k = max(top_k * 3, top_k)
        retrieved_chunks = self.store.search(query_embedding, initial_k)

        # Optional reranking
        reranked = self.reranker.rerank(question, retrieved_chunks, top_k)

        prompt = build_prompt(question, reranked)
        answer = self.generator.generate(prompt)

        return answer, reranked

