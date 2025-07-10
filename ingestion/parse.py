from unstructured.partition.pdf import partition_pdf

def parse_and_chunk_pdf(file_path):
    """
    Efficiently parse a PDF and yield chunks, each as a dict with 'title' and 'text' (paragraphs under that title).
    Designed for large-scale production use.
    """
    elements = partition_pdf(filename=file_path, strategy="fast")
    chunks = []
    current_chunk = {"title": None, "text": []}

    for el in elements:
        category = getattr(el, "category", None)
        text = getattr(el, "text", "").strip()
        if not text:
            continue
        if category == "Title":
            # Save previous chunk if it has content
            if current_chunk["title"] or current_chunk["text"]:
                chunk_text = "\n".join(current_chunk["text"])
                chunks.append({"title": current_chunk["title"], "text": chunk_text})
            # Start new chunk
            current_chunk = {"title": text, "text": []}
        elif category in ("NarrativeText", "ListItem", "BulletText"):
            current_chunk["text"].append(text)
    # Add last chunk
    if current_chunk["title"] or current_chunk["text"]:
        chunk_text = "\n".join(current_chunk["text"])
        chunks.append({"title": current_chunk["title"], "text": chunk_text})
    return chunks
