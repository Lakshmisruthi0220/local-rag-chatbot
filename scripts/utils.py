import os
from typing import List, Generator
import PyPDF2
from docx import Document

def load_txt(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def load_pdf(filepath: str) -> str:
    text = []
    with open(filepath, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return '\n'.join(text)

def load_docx(filepath: str) -> str:
    doc = Document(filepath)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> Generator[str, None, None]:
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        yield text[start:end]
        start += chunk_size - overlap

def list_documents(docs_dir: str) -> List[str]:
    documents = []
    for root, _, files in os.walk(docs_dir):
        for file in files:
            if file.lower().endswith(('.txt', '.pdf', '.docx')):
                documents.append(os.path.join(root, file))
    return documents
