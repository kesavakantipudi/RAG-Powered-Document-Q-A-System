from src.qa_pipeline import RAGPipeline


def test_rag_pipeline_returns_answer_and_sources():
    rag = RAGPipeline("data/raw")

    question = "What packages are being installed?"
    answer, sources = rag.answer(question)

    # Validate answer
    assert isinstance(answer, str)
    assert len(answer) > 0

    # Validate sources
    assert isinstance(sources, list)
    assert len(sources) > 0

    first_source = sources[0]
    assert "source" in first_source
    assert "page" in first_source
