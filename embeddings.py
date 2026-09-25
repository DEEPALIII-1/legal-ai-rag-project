import os
import pickle
import numpy as np

_EMBED_MODEL = None
_MODEL_NAME = "all-MiniLM-L6-v2"

def get_embedding_model():
    """Lazy loader for the SentenceTransformer embedding model."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            try:
                # Try loading from local HuggingFace cache first for fast offline startup
                _EMBED_MODEL = SentenceTransformer(_MODEL_NAME, local_files_only=True)
            except Exception:
                # Fallback to online fetch if cache missing
                _EMBED_MODEL = SentenceTransformer(_MODEL_NAME)
        except Exception as e:
            print(f"[Warning] SentenceTransformer load failed: {e}. Falling back to TF-IDF.")
            _EMBED_MODEL = None
    return _EMBED_MODEL


def embed_documents(docs, cache_file=None):
    """
    Computes dense normalized embeddings for a list of document strings.
    If cache_file is provided and exists, loads cached vectors.
    """
    if cache_file and os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
                if len(cached) == len(docs):
                    return cached
        except Exception:
            pass

    model = get_embedding_model()
    if model is not None:
        vectors = model.encode(docs, show_progress_bar=False, convert_to_numpy=True)
        # Normalize to unit length for fast cosine similarity via dot product
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms
    else:
        # Fallback to TF-IDF if neural model is unavailable
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        tfidf_mat = vec.fit_transform(docs).toarray()
        norms = np.linalg.norm(tfidf_mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = tfidf_mat / norms

    if cache_file:
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(vectors, f)
        except Exception:
            pass

    return vectors


def embed_query(query):
    """
    Encodes a single query string into a normalized dense vector.
    """
    model = get_embedding_model()
    if model is not None:
        q_vec = model.encode([query], show_progress_bar=False, convert_to_numpy=True)[0]
        norm = np.linalg.norm(q_vec)
        return q_vec / (norm if norm != 0 else 1.0)
    else:
        # Fallback dummy vector; sparse matching will handle retrieval
        return np.zeros(384)


def compute_cosine_similarity(query_vec, doc_vectors):
    """
    Computes cosine similarity between a normalized query vector and doc vectors.
    Returns 1D array of scores in range [0, 1].
    """
    if query_vec is None or doc_vectors is None or len(doc_vectors) == 0:
        return np.zeros(0)
    # Both query_vec and doc_vectors are normalized, so dot product is cosine similarity
    scores = np.dot(doc_vectors, query_vec)
    # Clip between 0 and 1
    return np.clip(scores, 0.0, 1.0)