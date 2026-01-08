from src.loaders import load_documents
from src.chunking import chunk_documents


def test_chunking_creates_chunks():
    docs = load_documents("data/raw")
    chunks = chunk_documents(docs)

    # Basic sanity checks
    assert isinstance(chunks, list)
    assert len(chunks) > 0

    first_chunk = chunks[0]

    # Validate chunk structure
    assert "chunk_id" in first_chunk
    assert "source" in first_chunk
    assert "page" in first_chunk
    assert "chunk_text" in first_chunk

    # Validate content
    assert isinstance(first_chunk["chunk_text"], str)
    assert len(first_chunk["chunk_text"]) > 0
