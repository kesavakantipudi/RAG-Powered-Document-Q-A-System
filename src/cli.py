from src.qa_pipeline import RAGPipeline


def main():
    print("RAG Document Q&A System")
    print("Type 'exit' to quit\n")

    rag = RAGPipeline("data/raw")

    NON_DOCUMENT_PATTERNS = [
        "about you",
        "who are you",
        "your name",
        "tell me about you",
        "something about you",
        "who am i",
    ]

    while True:
        question = input("Ask a question: ").strip()

        # Exit condition (MUST come first)
        if question.lower() == "exit":
            print("Goodbye!")
            break

        # Empty input guard
        if not question:
            print("Please enter a question or type 'exit' to quit.")
            continue

        # Very short / unclear input guard
        if len(question.split()) < 3:
            print("\nAnswer:")
            print(
                "Sorry, I can’t answer this question because it does not contain enough information."
            )
            print("-" * 50)
            continue

        # Non-document / personal question guard
        if any(p in question.lower() for p in NON_DOCUMENT_PATTERNS):
            print("\nAnswer:")
            print(
                "Sorry, I can’t answer this question because it is not related to the provided documents."
            )
            print("-" * 50)
            continue

        # Normal RAG flow
        answer, sources = rag.answer(question)

        print("\nAnswer:")
        print(answer)

        print("\nSources:")
        for i, s in enumerate(sources):
            print(f"[{i+1}] {s['source']} (page {s['page']})")

        print("-" * 50)


if __name__ == "__main__":
    main()
