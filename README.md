# RAG-Powered Document Q&A System

## Project Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system that answers user questions based strictly on the content of provided documents.
Instead of relying solely on a language model’s internal knowledge, the system retrieves **relevant document chunks using semantic search** and generates **factually grounded answers with source citations**.

The system supports PDF and TXT documents and demonstrates a complete, modular RAG pipeline suitable for real-world AI applications.

## Architecture Overview

The system follows a modular RAG architecture:

1. Documents are ingested from disk (PDF / TXT)
2. Text is split into overlapping chunks
3. Each chunk is converted into vector embeddings
4. FAISS performs semantic similarity search
5. Top-K relevant chunks are injected into a prompt
6. A language model generates an answer grounded in retrieved context
7. Source documents are cited in the final response

See [architecture.mmd](architecture.mmd) for a visual representation of the data flow (render with Mermaid). If you prefer a PNG, export this Mermaid file using the Mermaid CLI or VS Code Mermaid extension.

##  Technical Stack

- **Programming Language:** Python 3.10
- **Embedding Model:** all-MiniLM-L6-v2 (Sentence-Transformers)
- **Vector Store:** FAISS (IndexFlatL2)
- **LLM:** google/flan-t5-base (CPU)
- **Document Parsing:** pdfplumber
- **Environment:** Windows, CPU-only

## Project Structure


rag-document-qa/
├─ data/
│  └─ raw/
│     └─ sample.txt
├─ src/
|  ├─ __init__.py
│  ├─ loaders.py
│  ├─ chunking.py
│  ├─ embeddings.py
│  ├─ vector_store.py
│  ├─ prompt.py
│  ├─ generator.py
│  ├─ qa_pipeline.py
│  ├─ test_loader.py
│  ├─ test_chunking.py
│  ├─ test_embeddings.py
│  ├─ test_vector_store.py
│  └─ test_rag.py
├─ evaluation-report.md
├─ architecture.png
├─ requirements.txt
└─ README.md

##  Setup Instructions

### Create and activate virtual environment (Windows PowerShell)

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```


### Installing dependencies

```
pip install -r requirements.txt
```

## How to Run the System

Run each stage independently to verify correctness:

```
python src/test_loader.py
python src/test_chunking.py
python src/test_embeddings.py
python src/test_vector_store.py
python src/test_rag.py
```

Or start the interactive CLI:

```
python src/cli.py
```

By default, the system attempts to use a local FLAN-T5 model via `transformers`. If you prefer to skip downloading large models (e.g., for quick tests), set the following environment variable to use a lightweight fallback generator:

```
$env:RAG_USE_TRANSFORMERS = "0"
python src/test_rag.py
```


## Example Usage

**User Question:**

What packages are being installed?


**Answer:**

The document lists multiple Python packages being installed, including numpy, torch,
transformers, pdfplumber, and sentence-transformers. [1][2]


## Evaluation

The system was evaluated using **20 sample questions**.
Show Results:

- Accurate retrieval of relevant chunks
- Answers grounded in document content
- Correct and consistent source citations
- No hallucinations observed



##  Key Features

- Modular, extensible RAG pipeline
- Semantic search with FAISS
- Source-aware answer generation (with citations)
- CPU-only compatible
- Portfolio-ready AI engineering project

## Notes on Model Downloads

- `sentence-transformers` will download the embedding model (`all-MiniLM-L6-v2`) on first use.
- `transformers` will download `google/flan-t5-base` on first use if `RAG_USE_TRANSFORMERS` is enabled. Use the fallback generator by setting `$env:RAG_USE_TRANSFORMERS = "0"` to avoid large downloads.

