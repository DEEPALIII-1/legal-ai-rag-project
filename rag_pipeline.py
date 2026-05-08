from embeddings import get_embeddings, get_query_embedding
import numpy as np

docs_store = []
doc_vectors = None

def load_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().split("\n\n")

def store_docs(docs):
    global docs_store, doc_vectors
    docs_store = docs
    doc_vectors = get_embeddings(docs)

def search(query):
    query_vec = get_query_embedding(query)

    scores = (doc_vectors @ query_vec.T).toarray().flatten()

    top_index = np.argsort(scores)[-3:][::-1]

    return [docs_store[i] for i in top_index]