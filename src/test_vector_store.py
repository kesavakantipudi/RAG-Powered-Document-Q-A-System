from src.loaders import load_documents
from src.chunking import chunk_documents
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


def test_vector_store_search_returns_results():
    # Load and chunk documents
    docs = load_documents("data/raw")
    chunks = chunk_documents(docs)

    texts = [c["chunk_text"] for c in chunks]
    metadata = chunks

    assert len(texts) > 0

    # Create embeddings
    embedder = EmbeddingModel()
    embeddings = embedder.embed_texts(texts)

    # Build vector store
    store = VectorStore(embedding_dim=len(embeddings[0]))
    store.add(embeddings, metadata)

    # Perform search
    query = "What packages are being installed?"
    query_embedding = embedder.embed_texts([query])[0]

    results = store.search(query_embedding, top_k=2)

    # Validate search results
    assert isinstance(results, list)
    assert len(results) > 0

    first_result = results[0]
    assert "chunk_text" in first_result
    assert "source" in first_result
