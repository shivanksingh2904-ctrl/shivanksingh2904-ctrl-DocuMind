"""Ask questions about your PDFs. Run with:  streamlit run app.py"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

import pdf_processor
import rag

st.set_page_config(page_title="PDF Q&A", page_icon="📄", layout="wide")


def secret(name: str) -> str:
    try:
        value = st.secrets[name]
    except Exception:
        value = os.environ.get(name, "")
    return str(value) if value else ""


@st.cache_resource(show_spinner="Reading your PDFs...")
def build_index(files: tuple):
    """files = ((name, bytes), ...). Returns (index, problems)."""
    chunks, problems = [], []
    for name, data in files:
        try:
            found = pdf_processor.process_pdf(name, data)
        except Exception as err:
            problems.append(f"{name}: {err}")
            continue
        if found:
            chunks += found
        else:
            problems.append(f"{name}: no readable text (it may be a scanned image).")
    return rag.BM25Index(chunks), problems


api_key = secret("ANTHROPIC_API_KEY")

with st.sidebar:
    st.title("📄 PDF Q&A")
    uploads = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    top_k = st.slider("Passages to search", 2, 8, 4)
    st.caption("Mode: **AI answers**" if api_key else "Mode: **best-match passages** (add ANTHROPIC_API_KEY for AI answers)")
    if st.button("Clear chat"):
        st.session_state.pop("history", None)
        st.rerun()

st.title("Ask your documents")
if not uploads:
    st.info("Upload one or more PDFs in the sidebar, then ask a question.")
    st.stop()

index, problems = build_index(tuple((f.name, f.getvalue()) for f in uploads))
for p in problems:
    st.warning(p)
if not index.chunks:
    st.error("None of the uploaded PDFs contained readable text.")
    st.stop()
st.caption(f"Ready: {len(uploads)} file(s), {len(index.chunks)} searchable passages.")

history = st.session_state.setdefault("history", [])
for turn in history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("sources"):
            with st.expander("Sources"):
                for s in turn["sources"]:
                    st.markdown(f"**{s['source']} - page {s['page']}**")
                    st.write(s["text"])

question = st.chat_input("Ask a question about your PDFs")
if question:
    history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    hits = index.search(question, k=top_k)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            text, mode = rag.answer(question, hits, api_key)
        st.markdown(text)
        sources = [c for _, c in hits]
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"**{s['source']} - page {s['page']}**")
                    st.write(s["text"])
    history.append({"role": "assistant", "content": text, "sources": sources})
