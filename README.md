# Local RAG-Powered Chatbot (FAISS + Hugging Face) 

A simple Retrieval-Augmented Generation (RAG) chatbot that:
- Indexes local documents (txt, pdf) with SentenceTransformers embeddings.
- Stores embeddings in FAISS.
- Uses a local Hugging Face model (e.g., Mistral or LLaMA family) for generation.
- Provides a Streamlit UI for uploading queries.

## Features

- Upload and index documents in various formats (`.pdf`, `.docx`, `.txt`)
- Efficient document chunking and text preprocessing
- Vector-based retrieval using semantic embeddings
- Question answering over your private document data
- Easy to extend for additional document types or retrieval methods
- Local execution with no cloud dependencies

## Main Libraries Used

- **[python-docx](https://python-docx.readthedocs.io/en/latest/)** — For reading and parsing `.docx` documents  
- **[PyPDF2](https://pypi.org/project/PyPDF2/)** — For extracting text from PDF files  
- **[sentence-transformers](https://www.sbert.net/)** — To generate semantic embeddings for document chunks  
- **[faiss](https://github.com/facebookresearch/faiss)** — Fast similarity search library for vector retrieval  
- **[langchain](https://python.langchain.com/en/latest/)** — For chaining document retrieval and language model generation (if used)  
- **[streamlit](https://streamlit.io/)** — For building a simple interactive UI (optional)  

## Objective

Build a local, privacy-preserving chatbot that can answer questions based on your own collection of documents without sending data to external APIs. The system leverages semantic search and modern NLP techniques to provide precise answers by retrieving and processing relevant document fragments.

## Components Used

| Component        | Description                                                  |
|------------------|--------------------------------------------------------------|
| **Document Parsers**  | Extract text from different file formats (PDF, DOCX, TXT) using libraries like `PyPDF2` and `python-docx`. |
| **Text Chunking**     | Split large documents into smaller meaningful chunks for efficient indexing and retrieval. |
| **Embedding Model**   | Generate vector representations of text chunks using `sentence-transformers` models to capture semantic meaning. |
| **Vector Search Engine** | Use `faiss` for fast similarity search among embedding vectors to retrieve the most relevant chunks. |
| **Retrieval Pipeline** | Combine document parsing, chunking, embedding, and vector search to build a retrieval-augmented knowledge base. |
| **Question Answering Interface** | A UI built with `streamlit` or CLI interface to input questions and display answers by leveraging the retrieved document content. |
| **Utility Scripts**   | Scripts for indexing documents, managing data, and running the chatbot application. |


## Quick setup (local machine)

1. Clone repo:
```bash
git clone https://github.com/<your-username>/local-rag-chatbot.git
cd local-rag-chatbot



