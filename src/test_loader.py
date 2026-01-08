from src.loaders import load_documents


def test_load_documents_returns_data():
    docs = load_documents("data/raw")

    # Basic checks
    assert isinstance(docs, list)
    assert len(docs) > 0

    first_doc = docs[0]

    # Validate document structure
    assert "source" in first_doc
    assert "page" in first_doc
    assert "text" in first_doc

    # Validate content
    assert isinstance(first_doc["text"], str)
    assert len(first_doc["text"]) > 0
