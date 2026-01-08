import os
import pdfplumber

def load_txt(file_path):
    documents = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    documents.append({
        "text": text,
        "source": os.path.basename(file_path),
        "page": 1
    })
    return documents


def load_pdf(file_path):
    documents = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                documents.append({
                    "text": text,
                    "source": os.path.basename(file_path),
                    "page": i + 1
                })
    return documents


def load_documents(directory_path):
    all_documents = []

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        if filename.lower().endswith(".txt"):
            all_documents.extend(load_txt(file_path))

        elif filename.lower().endswith(".pdf"):
            all_documents.extend(load_pdf(file_path))

    return all_documents
