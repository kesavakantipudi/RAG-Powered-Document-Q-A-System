from src.loaders import load_documents
from src.chunking import chunk_documents
from src.embeddings import EmbeddingModel


def test_embedding_model_outputs_vectors():
    docs = load_documents("data/raw")
    chunks = chunk_documents(docs)

    texts = [c["chunk_text"] for c in chunks]
    assert len(texts) > 0

    embedder = EmbeddingModel()
    embeddings = embedder.embed_texts(texts)

    # Validate embeddings
    assert embeddings is not None
    assert len(embeddings) == len(texts)

    # Each embedding should be a vector
    assert len(embeddings[0]) > 0
