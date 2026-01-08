from src.loaders import load_documents
from src.chunking import chunk_documents
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.prompt import build_prompt
from src.generator import LocalGenerator

class RAGPipeline:
    def __init__(self, docs_path):
        # Load and chunk documents
        docs = load_documents(docs_path)
        chunks = chunk_documents(docs)

        self.texts = [c["chunk_text"] for c in chunks]
        self.metadata = chunks

        # Embeddings
        self.embedder = EmbeddingModel()
        embeddings = self.embedder.embed_texts(self.texts)

        # Vector store
        self.store = VectorStore(embedding_dim=embeddings.shape[1])
        self.store.add(embeddings, self.metadata)

        # Generator
        self.generator = LocalGenerator()

    def answer(self, question, top_k=3):
        query_embedding = self.embedder.embed_texts([question])[0]
        retrieved_chunks = self.store.search(query_embedding, top_k)

        prompt = build_prompt(question, retrieved_chunks)
        answer = self.generator.generate(prompt)

        
        
        return answer, retrieved_chunks

