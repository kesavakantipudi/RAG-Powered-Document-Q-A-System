def build_prompt(question, retrieved_chunks):
    context_blocks = []

    for i, chunk in enumerate(retrieved_chunks):
        block = (
            f"Source [{i+1}] ({chunk['source']}, page {chunk['page']}):\n"
            f"{chunk['chunk_text']}"
        )
        context_blocks.append(block)

    context = "\n\n".join(context_blocks)

    prompt = f"""
You are a document-based question answering system.

Use ONLY the information provided in the sources below.
Do NOT use any external knowledge.

If the answer cannot be found in the sources, respond with:
"Sorry, I can’t answer this question as the provided documents don’t contain the required information."

Always cite your sources using [1], [2], etc.

Sources:
{context}

Question:
{question}

Answer:
"""

    return prompt.strip()
