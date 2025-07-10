import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from ingestion.parse import parse_and_chunk_pdf
from ingestion.embed import embed_text
import numpy as np


DATA_DIR = "data"
BATCH_SIZE = 30  # Number of chunks to embed at once
MAX_WORKERS = 12 # Number of parallel PDF processors


def get_all_pdfs(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".pdf"):
                yield os.path.join(root, file)


def batch_embed_texts(texts):
    if not texts:
        return np.array([])
    model = embed_text.__globals__["get_model"]()  # Reuse model
    embeddings = model.encode(
        texts, show_progress_bar=False, normalize_embeddings=True
    )
    return np.array(embeddings)


def process_pdf(pdf_path):
    t0 = time.time()
    chunks = parse_and_chunk_pdf(pdf_path)
    chunk_texts = [chunk["text"] for chunk in chunks if chunk["text"]]
    chunk_titles = [chunk["title"] for chunk in chunks if chunk["text"]]
    embeddings = []
    # Batch embedding for speed
    for i in range(0, len(chunk_texts), BATCH_SIZE):
        batch = chunk_texts[i : i + BATCH_SIZE]
        batch_emb = batch_embed_texts(batch)
        embeddings.extend(batch_emb)
    t1 = time.time()
    return {
        "pdf_path": pdf_path,
        "num_chunks": len(chunk_texts),
        "titles": chunk_titles,
        "embeddings": embeddings,
        "time": t1 - t0,
    }


def main():
    # Eagerly load the model in the main thread before any parallel work
    embed_text.__globals__["get_model"]()
    pdf_paths = list(get_all_pdfs(DATA_DIR))
    print(f"Found {len(pdf_paths)} PDFs. Starting parallel processing...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_pdf, path) for path in pdf_paths]
        for future in as_completed(futures):
            result = future.result()
            print(f"\nProcessed: {result['pdf_path']}")
            print(f"  Chunks: {result['num_chunks']}, Time: {result['time']:.2f}s")
            for idx, (title, emb) in enumerate(
                zip(result["titles"], result["embeddings"])
            ):
                print(
                    f"    Chunk {idx+1}: Title: {title!r}, Embedding shape: {emb.shape}, First 5: {emb[:5]}"
                )
    t1 = time.time()
    print(f"\nTotal pipeline time: {t1-t0:.2f}s")


if __name__ == "__main__":
    main()
