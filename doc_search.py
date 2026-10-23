"""
Semantic search over large PDFs.

Flow: PDF -> raw text (PyPDF) -> chunks (LangChain splitter)
      -> embeddings (all-mpnet-base-v2) -> cosine-similarity search

Install:
    pip install pypdf sentence-transformers langchain-text-splitters numpy

Usage:
    python doc_search.py path/to/file.pdf
"""

import os
import pickle
import sys

import numpy as np
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-mpnet-base-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def extract_text(pdf_path):
    """Pull text out of each page. Returns a list of (page_number, text)."""
    reader = PdfReader(pdf_path)
    pages = []

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        # scanned pages or image-only pages come back empty, just skip them
        if text:
            pages.append((i, text))

    return pages


def split_pages(pages):
    """Break page text into overlapping chunks, remembering which page each came from."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = []
    for page_num, text in pages:
        for piece in splitter.split_text(text):
            chunks.append({"page": page_num, "text": piece})

    return chunks


class DocSearch:
    def __init__(self):
        print("Loading model (first run downloads it, be patient)...")
        self.model = SentenceTransformer(MODEL_NAME)
        self.chunks = []
        self.embeddings = None

    def load_pdf(self, pdf_path):
        cache_file = pdf_path + ".index.pkl"

        # don't re-embed a big PDF every single time
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            self.chunks = data["chunks"]
            self.embeddings = data["embeddings"]
            print(f"Loaded cached index ({len(self.chunks)} chunks)")
            return

        pages = extract_text(pdf_path)
        if not pages:
            raise ValueError("No extractable text found. Is this a scanned PDF?")

        self.chunks = split_pages(pages)
        print(f"Split into {len(self.chunks)} chunks, embedding now...")

        texts = [c["text"] for c in self.chunks]
        # normalizing means a plain dot product == cosine similarity
        self.embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        with open(cache_file, "wb") as f:
            pickle.dump({"chunks": self.chunks, "embeddings": self.embeddings}, f)

    def search(self, query, top_k=3):
        if self.embeddings is None:
            raise RuntimeError("Load a PDF first")

        q = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ q

        best = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "score": float(scores[i]),
                "page": self.chunks[i]["page"],
                "text": self.chunks[i]["text"],
            }
            for i in best
        ]


def main():
    if len(sys.argv) < 2:
        print("Usage: python doc_search.py <file.pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"Can't find {pdf_path}")
        sys.exit(1)

    engine = DocSearch()
    engine.load_pdf(pdf_path)

    print("\nReady. Ask something (type 'quit' to exit)\n")
    while True:
        query = input("> ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        for rank, hit in enumerate(engine.search(query), start=1):
            preview = hit["text"].replace("\n", " ")
            print(f"\n[{rank}] page {hit['page']}  (score {hit['score']:.2f})")
            print(preview[:400] + ("..." if len(preview) > 400 else ""))
        print()


if __name__ == "__main__":
    main()
