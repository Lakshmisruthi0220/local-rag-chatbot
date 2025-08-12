import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from scripts.utils import list_documents, load_txt, load_pdf, load_docx, chunk_text

DOCS_DIR = "data/docs"
INDEX_DIR = "data/faiss_index"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

def load_document(file_path):
    ext = file_path.lower().split('.')[-1]
    if ext == "txt":
        return load_txt(file_path)
    elif ext == "pdf":
        return load_pdf(file_path)
    elif ext == "docx":
        return load_docx(file_path)
    else:
        print(f"Unsupported file type: {file_path}")
        return ""

def main():
    os.makedirs(INDEX_DIR, exist_ok=True)
    docs = list_documents(DOCS_DIR)
    print(f"Found {len(docs)} documents in {DOCS_DIR}")

    all_chunks = []
    for doc_path in docs:
        print(f"Loading {doc_path}")
        text = load_document(doc_path)
        if not text.strip():
            print(f"Warning: No text found in {doc_path}")
            continue
        chunks = list(chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP))
        all_chunks.extend(chunks)

    print(f"Total chunks to index: {len(all_chunks)}")

    model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    embedder = SentenceTransformer(model_name)

    embeddings = embedder.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_path = os.path.join(INDEX_DIR, "faiss_index")
    meta_path = os.path.join(INDEX_DIR, "chunks.pkl")

    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"Index and metadata saved to {INDEX_DIR}")

if __name__ == "__main__":
    main()
