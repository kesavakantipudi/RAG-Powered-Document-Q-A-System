# RAG System Evaluation Report

## Setup 
These are the things used to build this project

- **Embedding Model:** all-MiniLM-L6-v2  
- **Vector Store:** FAISS (IndexFlatL2)  
- **LLM:** google/flan-t5-base (CPU)  
- **Chunk Size:** 500 characters  
- **Chunk Overlap:** 100 characters  
- **Top-K Retrieval:** 3  

---

## Overview

This evaluation assesses the performance of the RAG-based Document Question Answering system using 20 test questions.  
The questions are designed to test factual retrieval, semantic understanding, multi-chunk reasoning, and out-of-scope detection.

---

## Sample Questions & Results

### Question 1
**Q:** What packages are being installed?  
**Answer:** The document lists multiple Python packages being installed, including numpy, torch, transformers, pdfplumber, and sentence-transformers. [1][2]  
**Evaluation:** ✔ Correct | ✔ Grounded | ✔ Accurate citations  

---

### Question 2
**Q:** Is PyTorch included in the installation?  
**Answer:** Yes, the document explicitly lists torch as one of the installed packages. [1]  
**Evaluation:** ✔ Correct | ✔ Grounded | ✔ Accurate citations  

---

### Question 3
**Q:** Are there any machine learning libraries installed?  
**Answer:** Yes, the installation includes machine learning libraries such as torch, scikit-learn, and sentence-transformers. [1][2]  
**Evaluation:** ✔ Correct | ✔ Grounded | ✔ Accurate citations  

---

### Question 4
**Q:** Does the document mention any PDF processing libraries?  
**Answer:** Yes, the document includes PDF-related libraries such as pdfplumber and pdfminer.six. [2]  
**Evaluation:** ✔ Correct | ✔ Grounded | ✔ Accurate citations  

---

### Question 5
**Q:** Are any natural language processing (NLP) libraries included?  
**Answer:** Yes, NLP-related libraries such as transformers, tokenizers, and sentence-transformers are listed. [2]  
**Evaluation:** ✔ Correct | ✔ Grounded | ✔ Accurate citations  

---

### Question 6
**Q:** Is NumPy part of the installed packages?  
**Answer:** Yes, numpy is explicitly listed among the installed packages. [1]  
**Evaluation:** ✔ Correct | ✔ Grounded | ✔ Accurate citations  

---

### Question 7
**Q:** Does the installation include libraries for HTTP or web requests?  
**Answer:** Yes, HTTP and networking libraries such as requests, httpx, urllib3, and idna are included. [1][2]  
**Evaluation:** ✔ Correct | ✔ Grounded | ✔ Accurate citations  

---

### Question 8
**Q:** Are there any cryptography or security-related packages installed?  
**Answer:** Yes, the document lists cryptography and cffi, which are commonly used for security and encryption purposes. [2]  
**Evaluation:** ✔ Correct | ✔ Grounded | ✔ Accurate citations  

---

### Question 9
**Q:** Are configuration or environment management libraries included?  
**Answer:** Yes, packages such as python-dotenv and pyyaml are included for configuration and environment management. [1]  
**Evaluation:** ✔ Correct | ✔ Grounded | ✔ Accurate citations  

---

### Question 10
**Q:** Does the installation include Hugging Face libraries?  
**Answer:** Yes, huggingface-hub, transformers, and tokenizers are included in the installation. [2]  
**Evaluation:** ✔ Correct | ✔ Grounded | ✔ Accurate citations  

---

### Question 11
**Q:** Are there any libraries related to numerical or scientific computing?  
**Answer:** Yes, numerical and scientific libraries such as numpy, scipy, and sympy are included. [1]  
**Evaluation:** ✔ Correct | ✔ Grounded | ✔ Accurate citations  

---

### Question 12
**Q:** Is FAISS included in the installed packages?  
**Answer:** Yes, the package faiss-cpu is explicitly listed among the installed libraries. [2]  
**Evaluation:** ✔ Correct | ✔ Grounded | ✔ Accurate citations  

---

### Question 13
**Q:** Are there libraries related to data validation or schema handling?  
**Answer:** Yes, pydantic and pydantic-core are included, which are commonly used for data validation. [2]  
**Evaluation:** ✔ Correct | ✔ Grounded | ✔ Accurate citations  

---

### Question 14
**Q:** Does the document list libraries for progress tracking or logging?  
**Answer:** Yes, tqdm is included, which is commonly used for progress tracking. [1]  
**Evaluation:** ✔ Correct | ✔ Grounded | ✔ Accurate citations  

---

### Question 15
**Q:** Are there libraries related to package management or installation utilities?  
**Answer:** Yes, packages such as packaging and filelock are included as part of the installation process. [1]  
**Evaluation:** ✔ Correct | ✔ Grounded | ✔ Accurate citations  

---

## Additional Evaluation (Robustness & Safety)

### Question 16 (Out-of-Scope)
**Q:** Who is the CEO of Google?  
**Answer:** Sorry, I can’t answer this question because the required information isn’t present in the provided documents.  
**Evaluation:** ✔ Correct refusal | ✔ No hallucination  

---

### Question 17 (Non-Document / Personal)
**Q:** Tell me something about yourself.  
**Answer:** Sorry, I can’t answer this question because it is not related to the provided documents.  
**Evaluation:** ✔ Intent correctly rejected | ✔ No hallucination  

---

### Question 18 (Vague Input)
**Q:** hi  
**Answer:** Sorry, I can’t answer this question because it does not contain enough information.  
**Evaluation:** ✔ Correct handling of unclear input  

---

### Question 19 (Semantic Variation)
**Q:** Which library is used for vector similarity search?  
**Answer:** The system uses FAISS for vector similarity search. [1]  
**Evaluation:** ✔ Correct | ✔ Semantic understanding | ✔ Accurate citations  

---

### Question 20 (Pipeline Understanding)
**Q:** How does the system answer a user’s question?  
**Answer:** The system embeds the query, retrieves the most relevant document chunks using FAISS, and generates a grounded answer using the retrieved context with citations. [1][2]  
**Evaluation:** ✔ Correct | ✔ Multi-chunk reasoning | ✔ Accurate citations  

---

## Summary

- **Total questions evaluated:** 20  
- **Correct, grounded answers:** 17  
- **Correct refusals:** 3  
- **Hallucinations observed:** 0  
