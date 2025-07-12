import os
import time
import io
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from ingestion.parse import parse_and_chunk_pdf
from ingestion.embed import embed_text
import numpy as np
import requests
import re
from urllib.parse import urlparse, parse_qs
import torch

DATA_DIR = "data"
BATCH_SIZE = 64  # Optimized for GPU processing
MAX_WORKERS = 8  # Reduced for better GPU utilization

# Google Drive settings
DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1VFKa-S9toPp6kIeHH6T3h8hI51gahn0A"


def get_all_pdfs(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".pdf"):
                yield os.path.join(root, file)


def batch_embed_texts(texts):
    if not texts:
        return np.array([])
    
    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = embed_text.__globals__["get_model"]()  # Reuse model
    
    embeddings = model.encode(
        texts, 
        show_progress_bar=False, 
        normalize_embeddings=True,
        device=device  # Use GPU for encoding
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


def extract_folder_id(drive_url):
    """Extract folder ID from Google Drive URL"""
    if '/folders/' in drive_url:
        return drive_url.split('/folders/')[-1].split('?')[0]
    return None


def get_pdfs_from_drive(folder_id):
    """Get list of PDF files from Google Drive folder using public access"""
    try:
        # This is a simplified approach - for production, use Google Drive API
        print(f"Note: For Google Drive access, you'll need to:")
        print("1. Make the folder publicly accessible, or")
        print("2. Set up Google Drive API credentials")
        print("3. For now, manually download PDFs to data folder")
        return []
    except Exception as e:
        print(f"Error accessing Google Drive: {e}")
        return []


def download_pdf_from_drive(file_id, filename):
    """Download PDF from Google Drive (simplified version)"""
    try:
        # This would need proper Google Drive API implementation
        # For now, return None to indicate local processing
        return None
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return None


def get_all_pdf_sources():
    """Get PDF sources from both local directory and Google Drive"""
    sources = []
    
    # 1. Get local PDFs
    local_pdfs = list(get_all_pdfs(DATA_DIR))
    for pdf_path in local_pdfs:
        sources.append({
            'type': 'local',
            'path': pdf_path,
            'name': os.path.basename(pdf_path)
        })
    
    # 2. Get Google Drive PDFs (if configured)
    if DRIVE_FOLDER_URL:
        folder_id = extract_folder_id(DRIVE_FOLDER_URL)
        if folder_id:
            drive_pdfs = get_pdfs_from_drive(folder_id)
            for pdf_info in drive_pdfs:
                sources.append({
                    'type': 'drive',
                    'file_id': pdf_info.get('id'),
                    'name': pdf_info.get('name'),
                    'path': None
                })
    
    return sources


def process_pdf_source(source):
    """Process PDF from either local file or Google Drive"""
    t0 = time.time()
    
    if source['type'] == 'local':
        # Process local file
        pdf_path = source['path']
        chunks = parse_and_chunk_pdf(pdf_path)
        source_name = pdf_path
    
    elif source['type'] == 'drive':
        # Download and process from Google Drive
        temp_file = download_pdf_from_drive(source['file_id'], source['name'])
        if temp_file is None:
            print(f"Skipping {source['name']} - Google Drive access not configured")
            return None
        
        chunks = parse_and_chunk_pdf(temp_file)
        source_name = f"Drive: {source['name']}"
        
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    else:
        print(f"Unknown source type: {source['type']}")
        return None
    
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
        "source": source_name,
        "type": source['type'],
        "name": source['name'],
        "num_chunks": len(chunk_texts),
        "titles": chunk_titles,
        "embeddings": embeddings,
        "time": t1 - t0,
    }


def main():
    # Check GPU availability and system info
    print("=== System Information ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory // 1024**3} GB")
    else:
        print("Running on CPU only")
    print(f"Batch size: {BATCH_SIZE}, Workers: {MAX_WORKERS}")
    print("=" * 30)
    
    # Eagerly load the model in the main thread before any parallel work
    embed_text.__globals__["get_model"]()
    
    # Get all PDF sources (local + Google Drive)
    pdf_sources = get_all_pdf_sources()
    local_count = sum(1 for s in pdf_sources if s['type'] == 'local')
    drive_count = sum(1 for s in pdf_sources if s['type'] == 'drive')
    
    print(f"Found {len(pdf_sources)} PDFs total:")
    print(f"  - Local files: {local_count}")
    print(f"  - Google Drive files: {drive_count}")
    print("Starting parallel processing...")
    
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_pdf_source, source) for source in pdf_sources]
        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue
                
            print(f"\nProcessed: {result['source']}")
            print(f"  Type: {result['type']}, Chunks: {result['num_chunks']}, Time: {result['time']:.2f}s")
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
