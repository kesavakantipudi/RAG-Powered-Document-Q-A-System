# chunking.py
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks


def chunk_documents(documents, chunk_size=500, overlap=100):
    all_chunks = []

    for doc in documents:
        text_chunks = chunk_text(doc["text"], chunk_size, overlap)

        for idx, chunk in enumerate(text_chunks):
            all_chunks.append({
                "chunk_text": chunk,
                "source": doc["source"],
                "page": doc["page"],
                "chunk_id": f"{doc['source']}_p{doc['page']}_c{idx}"
            })

    return all_chunks
