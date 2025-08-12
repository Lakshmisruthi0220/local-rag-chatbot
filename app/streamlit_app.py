import os
import pickle
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, logging

logging.set_verbosity_error()

INDEX_DIR = r"C:\Users\hp\Downloads\local-rag-chatbot\data\faiss_index"

@st.cache_resource(show_spinner=False)
def load_index(index_dir=INDEX_DIR):
    idx_path = os.path.join(index_dir, "faiss_index")
    meta_path = os.path.join(index_dir, "chunks.pkl")

    if not os.path.exists(idx_path) or not os.path.exists(meta_path):
        st.error("Index not found. Run the indexing script first (scripts/index_docs.py).")
        st.stop()
    index = faiss.read_index(idx_path)
    with open(meta_path, "rb") as f:
        md = pickle.load(f)
    return index, md

@st.cache_resource(show_spinner=False)
def load_embedder(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

@st.cache_resource(show_spinner=False)
def load_generator(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    gen = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if st.session_state.get("use_gpu", False) else -1,
    )
    return gen

def retrieve(query: str, index, md_chunks, embedder, top_k=5):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    chunks = []
    seen = set()
    for idx in I[0]:
        if idx < 0 or idx >= len(md_chunks):
            continue
        chunk = md_chunks[idx]
        if chunk not in seen:
            chunks.append(chunk)
            seen.add(chunk)
    return chunks
def make_prompt(context_chunks, query):
    context = "\n\n".join(context_chunks)
    return f"""Answer the question using only the information below.
If the answer is not contained in the text, say "The document does not contain information about that."

Context:
{context}

Question: {query}

Answer:"""

def main():
    st.set_page_config(page_title="Local RAG Chatbot", layout="wide")
    st.title("Local RAG-Powered Chatbot (FAISS + Local LLM)")
    st.markdown("Upload docs to `data/docs/` and run `python scripts/index_docs.py` to build the index.")

    col1, col2 = st.columns([1, 3])
    with col1:
        st.header("Index / Model")
        st.text(f"Index dir: {INDEX_DIR}")
        if st.button("Reload index"):
            load_index()
            st.success("Index reloaded (cached).")

        model_name = st.text_input("Generation model (Hugging Face repo)", value="google/flan-t5-base")

        if st.button("Load generator"):
            with st.spinner("Loading generator (may take some time)..."):
                try:
                    gen = load_generator(model_name)
                    st.session_state["generator"] = gen
                    st.session_state["use_gpu"] = True  # set True if GPU available
                    st.success(f"Loaded {model_name}")
                except Exception as e:
                    st.error(f"Error loading model: {e}")

    with col2:
        index, md = load_index()
        embedder = load_embedder()
        query = st.text_input("Ask a question about your documents:")
        top_k = st.slider("Retrieval top_k", 1, 10, 3)

        if st.button("Search & Answer") and query.strip():
            with st.spinner("Retrieving relevant documents..."):
                retrieved_chunks = retrieve(query, index, md, embedder, top_k=top_k)
            if not retrieved_chunks:
                st.warning("No relevant information found in your documents for this query.")
            else:
                prompt = make_prompt(retrieved_chunks, query)

                st.subheader("Retrieved context (top chunks)")
                for i, c in enumerate(retrieved_chunks):
                    st.markdown(f"**Chunk {i+1}**")
                    st.write(c[:1000] + ("..." if len(c) > 1000 else ""))

                if "generator" not in st.session_state:
                    st.error("Generator not loaded. Go to left panel and click 'Load generator'.")
                else:
                    gen = st.session_state["generator"]
                    with st.spinner("Generating answer..."):
                        output = gen(prompt, max_length=256, do_sample=True, top_p=0.9, temperature=0.7)
                    answer = output[0]["generated_text"]
                    st.subheader("Answer")
                    st.write(answer)

if __name__ == "__main__":
    main()
