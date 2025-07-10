from sentence_transformers import SentenceTransformer
import numpy as np

# Load the model once at module level for efficiency
_model = None

def get_model(model_name="all-MiniLM-L6-v2"):
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model

def embed_text(text, model_name="all-MiniLM-L6-v2"):
    """
    Generate a dense embedding for the input text using sentence-transformers.
    Returns a numpy array.
    """
    model = get_model(model_name)
    # SentenceTransformer expects a list of strings
    embedding = model.encode([text], show_progress_bar=False, normalize_embeddings=True)
    return np.array(embedding[0])
