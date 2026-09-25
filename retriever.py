import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from embeddings import embed_documents, embed_query, compute_cosine_similarity

# Common stopwords to exclude from keyword boost
STOPWORDS = {
    "and", "or", "the", "in", "on", "at", "to", "for", "with", "by", "from", "of", "an", "as",
    "is", "was", "are", "were", "it", "its", "that", "this", "my", "me", "i", "we", "you", "he",
    "she", "they", "them", "his", "her", "their", "took", "had", "has", "have", "be", "been",
    "being", "do", "does", "did", "doing", "a", "about", "above", "after", "again", "against",
    "all", "am", "any", "can", "could", "our", "ours", "out", "over", "own", "same", "so",
    "than", "too", "very", "s", "t", "just", "don", "should", "now", "what", "which", "who",
    "whom", "why", "how", "where", "when", "some", "someone", "give", "gives", "getting", "got"
}

# Legal intent dictionary for intelligent query expansion (root matching)
LEGAL_EXPANSIONS = {
    "chequ": ["section 138", "dishonour of cheque", "bounced cheque", "legal notice 30 days", "ni act 138"],
    "check": ["cheque bounce", "section 138", "dishonour", "insufficient funds"],
    "bounc": ["section 138", "cheque bounce", "dishonour", "15 days demand notice"],
    "upi": ["cyber fraud", "it act 66d", "1930 helpline", "online banking fraud", "cybercrime.gov.in"],
    "gpay": ["upi fraud", "cyber fraud", "it act 66d", "1930", "phishing"],
    "phonepe": ["upi fraud", "cyber fraud", "it act 66d", "1930"],
    "paytm": ["upi fraud", "cyber fraud", "it act 66d", "1930"],
    "scam": ["cheating", "fraud", "bns 318", "ipc 420", "cyber fraud", "it act 66d"],
    "fraud": ["cheating", "bns 318", "ipc 420", "cyber crime", "restitution"],
    "cheat": ["cheating", "bns 318", "ipc 420", "fraudulent inducement", "breach of trust 406"],
    "fir": ["first information report", "bnss 173", "crpc 154", "zero fir", "lalita kumari", "section 156 3"],
    "polic": ["fir registration", "zero fir", "crpc 154", "bnss 173", "superintendent of police 154 3", "magistrate 156 3", "d k basu"],
    "arrest": ["arrest rights", "section 41a", "notice of appearance", "d k basu", "anticipatory bail crpc 438 bnss 482"],
    "bail": ["anticipatory bail", "regular bail", "crpc 438", "crpc 439", "bnss 482", "default bail 167 2", "satender kumar antil"],
    "refund": ["consumer protection act", "defective product", "deficiency of service", "edaakhil", "1915 helpline"],
    "amazon": ["consumer protection act", "e-commerce refund", "defective product", "deficiency of service"],
    "flipkart": ["consumer protection act", "e-commerce refund", "defective product", "deficiency of service"],
    "tenant": ["landlord", "security deposit", "eviction", "model tenancy act", "transfer of property act 106", "order 37 cpc"],
    "landlord": ["tenant", "security deposit withholding", "illegal eviction", "essential supplies disconnection", "model tenancy act"],
    "deposit": ["security deposit refund", "landlord tenant dispute", "summary suit order 37 cpc"],
    "rent": ["rent agreement", "eviction notice", "model tenancy act", "security deposit refund"],
    "salari": ["unpaid salary", "wrongful termination", "payment of wages act", "labor commissioner", "severance pay"],
    "salary": ["unpaid salary", "wrongful termination", "payment of wages act", "labor commissioner", "severance pay"],
    "fire": ["wrongful termination", "notice period pay", "severance", "labor court", "industrial disputes act"],
    "fired": ["wrongful termination", "notice period pay", "severance", "labor court", "industrial disputes act"],
    "terminat": ["wrongful termination", "retrenchment compensation", "labor court", "severance"],
    "posh": ["sexual harassment workplace", "internal complaints committee icc", "posh act 2013", "90 days inquiry"],
    "harass": ["domestic violence pwdva", "posh act", "cyber stalking", "criminal intimidation bns 351"],
    "domest": ["domestic violence", "pwdva section 12", "protection order", "residence order", "maintenance", "bns 85", "ipc 498a"],
    "husband": ["domestic violence", "maintenance crpc 125 bnss 144", "pwdva", "divorce 13b", "stridhan"],
    "wife": ["maintenance crpc 125", "divorce mutual consent 13b", "domestic violence", "child custody"],
    "divorc": ["mutual consent divorce 13b", "alimony", "maintenance crpc 125", "cooling off waiver", "family court"],
    "accid": ["motor vehicles act 166", "mact claim", "rash driving bns 281", "hit and run compensation", "good samaritan"],
    "crash": ["motor vehicles act 166", "mact claim", "rash driving bns 281", "hit and run compensation"],
    "photo": ["it act 66e", "it act 67a", "privacy violation", "revenge porn", "cyber blackmail", "takedown 24 hours"],
    "video": ["it act 66e", "it act 67a", "blackmail", "cyber stalking", "sextortion"],
    "nude": ["it act 66e", "it act 67a", "blackmail", "revenge porn", "stopncii org", "cybercrime helpline 1930"],
    "blackmail": ["extortion bns 308", "ipc 384", "it act 66e", "sextortion", "loan app harassment"],
    "extort": ["bns 308", "ipc 384", "loan app harassment", "criminal intimidation 351"],
    "loan": ["predatory lending", "recovery agent harassment", "sachet rbi", "extortion 308", "cybercrime 1930"],
    "rti": ["right to information act 2005", "section 6", "30 days deadline", "48 hours life liberty", "first appeal", "cic"],
    "defam": ["criminal defamation bns 356", "ipc 499 500", "civil suit for damages", "cease and desist notice", "social media post"],
    "contract": ["breach of contract", "section 73 indian contract act", "unpaid invoice", "specific performance", "commercial court"],
    "bns": ["bharatiya nyaya sanhita", "new criminal laws 2024", "ipc cross reference", "bns 318 420"],
    "ipc": ["indian penal code", "bns cross reference", "ipc 420 bns 318"],
    "420": ["bns 318", "ipc 420", "cheating", "fraud"],
    "138": ["section 138 ni act", "cheque bounce", "dishonour of cheque"]
}


def expand_query(query):
    """
    Intelligently enriches a user query with legal keywords, statutory citations, and synonyms.
    """
    query_clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', query.lower())
    words = query_clean.split()
    expansions = []
    
    for w in words:
        if w in STOPWORDS:
            continue
        # Direct word match
        if w in LEGAL_EXPANSIONS:
            expansions.extend(LEGAL_EXPANSIONS[w])
        else:
            # Stem prefix match (e.g. cheated -> cheat, bounced -> bounc)
            for key, exp_list in LEGAL_EXPANSIONS.items():
                if w.startswith(key) or key.startswith(w):
                    expansions.extend(exp_list)
                    break
            
    # Check multi-word phrases
    for phrase in ["hit and run", "cheque bounce", "loan app", "wrongful termination", "unpaid salary", "domestic violence", "cyber fraud"]:
        if phrase in query_clean and phrase in LEGAL_EXPANSIONS:
            expansions.extend(LEGAL_EXPANSIONS[phrase])
            
    if expansions:
        unique_expansions = list(dict.fromkeys(expansions))
        return f"{query} {' '.join(unique_expansions[:6])}"
    return query


class LegalHybridRetriever:
    def __init__(self, documents, cache_path="data/.cache_embeddings.pkl"):
        """
        Initializes the Hybrid Retriever with dense vectors and sparse TF-IDF index.
        """
        self.raw_documents = documents
        self.doc_texts = []
        self.doc_metadata = []
        self.cache_path = cache_path

        self._prepare_corpus()
        self._build_sparse_index()
        self._build_dense_index()

    def _prepare_corpus(self):
        self.doc_texts = []
        self.doc_metadata = []

        for i, item in enumerate(self.raw_documents):
            if isinstance(item, dict):
                title = item.get("title", "")
                cat = item.get("category", "")
                act = item.get("act", "")
                sections = item.get("sections", "")
                summary = item.get("summary", "")
                elements = " ".join(item.get("key_elements", []))
                penalties = item.get("penalties_remedies", "")
                cases = " ".join(item.get("landmark_cases", []))
                steps = " ".join(item.get("practical_steps", []))
                keywords = " ".join(item.get("keywords", []))

                full_search_text = f"{title}. Category: {cat}. Act: {act}. Sections: {sections}. Summary: {summary}. Elements: {elements}. Penalties: {penalties}. Precedents: {cases}. Action Steps: {steps}. Keywords: {keywords}"
                self.doc_texts.append(full_search_text)
                self.doc_metadata.append(item)
            else:
                text = str(item).strip()
                self.doc_texts.append(text)
                self.doc_metadata.append({
                    "id": f"DOC-{i+1}",
                    "title": text.split("\n")[0][:80],
                    "category": "General Legal Reference",
                    "act": "Indian Legal Framework",
                    "sections": "Applicable Statutes",
                    "summary": text,
                    "key_elements": [],
                    "penalties_remedies": "Refer to applicable law",
                    "landmark_cases": [],
                    "practical_steps": [text],
                    "relevant_authorities": "Legal & Judicial Authorities",
                    "risk_level": "Medium",
                    "keywords": []
                })

    def _build_sparse_index(self):
        """Constructs TF-IDF sparse lexical search matrix."""
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words='english'
        )
        self.sparse_matrix = self.vectorizer.fit_transform(self.doc_texts)

    def _build_dense_index(self):
        """Builds or loads cached dense embeddings."""
        self.dense_vectors = embed_documents(self.doc_texts, cache_file=self.cache_path)

    def search(self, query, top_k=3, dense_weight=0.55, sparse_weight=0.45, category_filter=None):
        """
        Executes hybrid search combining dense semantic vectors and sparse lexical BM25/TF-IDF.
        """
        if not query or not query.strip() or len(self.doc_texts) == 0:
            return []

        # 1. Expand query with legal synonyms & sections
        enriched_query = expand_query(query)

        # 2. Dense Semantic Retrieval
        q_dense = embed_query(query)
        dense_scores = compute_cosine_similarity(q_dense, self.dense_vectors)

        # 3. Sparse Lexical Retrieval
        q_sparse = self.vectorizer.transform([enriched_query])
        sparse_scores = (self.sparse_matrix @ q_sparse.T).toarray().flatten()
        max_sparse = np.max(sparse_scores) if len(sparse_scores) > 0 else 0
        if max_sparse > 0:
            sparse_scores = sparse_scores / max_sparse

        # 4. Keyword / Section Exact Match Boost (Strict, stopword-free whole-word regex)
        exact_boost = np.zeros(len(self.doc_texts))
        raw_words = re.findall(r'\b[a-zA-Z0-9]+\b', query.lower())
        filtered_words = [w for w in raw_words if w not in STOPWORDS and len(w) >= 3]

        for i, meta in enumerate(self.doc_metadata):
            meta_keywords = [k.lower() for k in meta.get("keywords", [])]
            meta_text_lower = self.doc_texts[i].lower()
            sections_str = meta.get("sections", "").lower()

            for w in filtered_words:
                # Exact whole-word match in sections or keywords
                if any(w == k or w in k.split() for k in meta_keywords):
                    exact_boost[i] += 0.12
                # Section match (e.g. 420, 138, 498a, 66d, bns, crpc)
                if re.search(r'\b' + re.escape(w) + r'\b', sections_str):
                    exact_boost[i] += 0.20

        # 5. Hybrid Weighted Score
        hybrid_scores = (dense_weight * dense_scores) + (sparse_weight * sparse_scores) + exact_boost

        # 6. Apply Category Filter if specified
        if category_filter and category_filter != "All":
            for i, meta in enumerate(self.doc_metadata):
                if meta.get("category") != category_filter:
                    hybrid_scores[i] = -1.0

        # 7. Select Top-K
        top_indices = np.argsort(hybrid_scores)[::-1]
        
        results = []
        for idx in top_indices:
            score = float(hybrid_scores[idx])
            if score <= 0.08 and len(results) >= 1:
                break
            meta = dict(self.doc_metadata[idx])
            meta["score"] = round(min(score, 1.0), 3)
            meta["dense_score"] = round(float(dense_scores[idx]), 3)
            meta["sparse_score"] = round(float(sparse_scores[idx]), 3)
            meta["text"] = self.doc_texts[idx]
            results.append(meta)
            if len(results) >= top_k:
                break

        return results

    def add_custom_documents(self, new_docs):
        """
        Dynamically adds custom documents (e.g. uploaded PDFs/texts) to the index.
        """
        for item in new_docs:
            self.raw_documents.append(item)
        self._prepare_corpus()
        self._build_sparse_index()
        self.dense_vectors = embed_documents(self.doc_texts, cache_file=None)
