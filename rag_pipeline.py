import os
import json
from retriever import LegalHybridRetriever
from generator import generate_rag_response, assess_risk, draft_legal_notice
from ingestion import process_uploaded_document

# Global retriever singleton
_GLOBAL_RETRIEVER = None
docs_store = []
doc_vectors = None


def load_file(path):
    """Loads a text file split by double newlines (backward compatible)."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read().strip()
        if not content:
            return []
        return [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]


def load_legal_knowledge_base():
    """
    Loads both structured legal JSON database and raw text dataset files.
    """
    all_docs = []

    # 1. Load structured JSON database
    json_path = os.path.join(os.path.dirname(__file__), "data", "legal_database.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                json_docs = json.load(f)
                all_docs.extend(json_docs)
        except Exception as e:
            print(f"[Warning] Failed loading legal_database.json: {e}")

    # 2. Also load text files if they contain additional information
    laws_path = os.path.join(os.path.dirname(__file__), "data", "legal_laws.txt")
    cases_path = os.path.join(os.path.dirname(__file__), "data", "court_cases.txt")
    
    text_laws = load_file(laws_path)
    text_cases = load_file(cases_path)

    for item in text_laws + text_cases:
        all_docs.append(item)

    return all_docs


def get_pipeline():
    """Returns initialized LegalHybridRetriever instance."""
    global _GLOBAL_RETRIEVER
    if _GLOBAL_RETRIEVER is None:
        docs = load_legal_knowledge_base()
        cache_path = os.path.join(os.path.dirname(__file__), "data", ".cache_embeddings.pkl")
        _GLOBAL_RETRIEVER = LegalHybridRetriever(docs, cache_path=cache_path)
    return _GLOBAL_RETRIEVER


def store_docs(docs):
    """
    Updates the pipeline documents in memory (backward compatible).
    """
    global _GLOBAL_RETRIEVER, docs_store
    docs_store = docs
    _GLOBAL_RETRIEVER = LegalHybridRetriever(docs, cache_path=None)


def search(query, top_k=3, category_filter=None):
    """
    Backward-compatible search returning top matching snippets or document dictionaries.
    """
    pipeline = get_pipeline()
    results = pipeline.search(query, top_k=top_k, category_filter=category_filter)
    # Return formatted strings or summaries
    output = []
    for r in results:
        title = r.get("title", "")
        summary = r.get("summary", "")
        sections = r.get("sections", "")
        output.append(f"[{title}] (Sections: {sections})\n{summary}")
    return output


def query_rag(query, category=None, top_k=3, api_key=None, provider="gemini"):
    """
    Complete end-to-end RAG pipeline execution:
    1. Hybrid retrieval (Dense Semantic + Sparse Lexical + Exact legal boost)
    2. Multi-factor legal risk assessment
    3. Grounded answer generation via external LLM or smart local legal synthesizer
    """
    pipeline = get_pipeline()
    retrieved_docs = pipeline.search(query, top_k=top_k, category_filter=category)
    
    # Generate structured answer
    answer, risk_info = generate_rag_response(
        query=query,
        retrieved_docs=retrieved_docs,
        api_key=api_key,
        provider=provider
    )

    return {
        "query": query,
        "answer": answer,
        "risk_info": risk_info,
        "retrieved_docs": retrieved_docs
    }


def add_document(file_name, file_bytes):
    """
    Ingests and indexes an uploaded PDF or TXT document into the active retriever.
    """
    pipeline = get_pipeline()
    new_entries = process_uploaded_document(file_name, file_bytes)
    if new_entries:
        pipeline.add_custom_documents(new_entries)
    return len(new_entries)