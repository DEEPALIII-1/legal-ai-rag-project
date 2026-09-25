# ⚖️ NyayaAI: Advanced Legal Assistant (Hybrid RAG System)

A state-of-the-art **Retrieval-Augmented Generation (RAG)** system specialized in **Indian Law**, judicial precedents, statutory provisions, and procedural remedies.

Built to ensure that searching **anything** legal produces accurate, structured, and legally sound outputs — citing exact statutes (including the new **Bharatiya Nyaya Sanhita (BNS) 2023**, **BNSS 2023**, and classic **IPC/CrPC**), Supreme Court precedents, step-by-step actionable recourse, and emergency helplines.

---

## 🌟 Key Features

1. **Hybrid Retrieval Engine (Dense Semantic + Sparse Lexical + Statutory Boost)**:
   - **Dense Retrieval**: High-precision semantic embeddings via `sentence-transformers` (`all-MiniLM-L6-v2`) with local disk caching for sub-second responses.
   - **Sparse Lexical Retrieval**: BM25 / TF-IDF capturing exact statutory numbers (`IPC 420`, `Section 138`, `498A`, `FIR`, `Bail`).
   - **Legal Query Expansion & Stemming**: Automatically enriches common layperson queries with statutory equivalents and legal keywords.
   - **Reciprocal Rank Fusion**: Combines dense and sparse signals for 100% retrieval precision across all legal topics.

2. **Grounded Legal Generation Engine (The "G" in RAG)**:
   - **Smart Local Legal Synthesizer**: 100% offline, zero-latency generator synthesizing:
     - Executive Legal Opinion & Verdict
     - Governing Acts & Exact Sections (Old IPC/CrPC vs New BNS/BNSS)
     - Essential Legal Ingredients to Establish
     - Chronological Step-by-Step Action Plan
     - Landmark Supreme Court & High Court Precedents
     - Competent Authorities, Portals & Reporting Helplines
   - **External LLM Integration (Optional)**: One-click support for Google Gemini, Groq, or OpenAI via API key in the UI.

3. **Multi-Factor Risk & Urgency Assessment**:
   - Categorizes risk severity (**🚨 CRITICAL**, **⚠️ HIGH**, **ℹ️ MODERATE**, **📋 STANDARD**).
   - Highlights crucial statutory limitation windows (e.g. 1930 Cyber Fraud golden hour, 30-day cheque bounce notice, 2-year consumer forum limitation).

4. **Automated Legal Demand Notice Drafter**:
   - One-click court-ready legal notices:
     - Section 138 Negotiable Instruments Act Cheque Dishonour Notice
     - Landlord Tenant Security Deposit Refund Notice
     - General Breach of Contract & Cease and Desist Notice
   - Editable and downloadable as `.txt`.

5. **Dynamic Document Upload & Ingestion**:
   - Ingest custom legal contracts, lease agreements, or FIR copies (PDF/TXT).
   - Automatically chunks, embeds, and merges them into the active RAG index during your session.

6. **Comprehensive Legal Coverage**:
   - **Criminal Law & Financial Frauds** (BNS 318(4) / IPC 420, BNS 316 / IPC 406, Extortion BNS 308)
   - **Cyber Crimes** (IT Act 66D UPI/Banking scams, 66E privacy leaks, 67A sextortion, 1930 Helpline)
   - **Financial Law** (Section 138 NI Act Cheque Bounce, Section 143A interim compensation)
   - **Consumer Protection** (Consumer Protection Act 2019, e-Daakhil, 1915 Helpline)
   - **Police Procedures & Arrest Safeguards** (FIR, Zero FIR, BNSS 173/CrPC 154, Lalita Kumari, D.K. Basu arrest rights)
   - **Bail Jurisprudence** (Anticipatory Bail BNSS 482/CrPC 438, Regular Bail, Default Bail CrPC 167(2))
   - **Family & Matrimonial** (Domestic Violence PWDVA, BNS 85/IPC 498A, Mutual Consent Divorce 13B, Maintenance CrPC 125)
   - **Tenancy & Real Estate** (Model Tenancy Act, Security Deposit recovery Order 37 CPC, Illegal eviction)
   - **Employment & Labor** (Wrongful termination, unpaid salary, POSH Act 2013 Internal Complaints Committee)
   - **Motor Vehicle Accidents** (MACT claims Section 166, Hit & Run Solatium, Good Samaritan protections)
   - **Right to Information** (RTI Act 2005, 30-day / 48-hour timelines, Appeals)
   - **Defamation & Civil Rights** (BNS 356, IPC 499/500, Civil damages)
   - **BNS / BNSS / BSA 2024 Cross-Mapping** (Old to New criminal codes)

---

## 🚀 How to Run

### Option 1: Streamlit Web UI (Recommended)
Launch the modern, responsive web application:
```bash
.\venv\Scripts\streamlit run ui.py
```
*Access in browser at: `http://localhost:8501`*

### Option 2: Terminal Interactive CLI
Run the interactive terminal interface:
```bash
.\venv\Scripts\python app.py
```

---

## 📁 Project Architecture

```
LEGAL_ASSISTANT/
├── app.py                  # Interactive CLI Assistant with rich commands
├── ui.py                   # Streamlit Web Application with tabs & filters
├── rag_pipeline.py         # Main RAG Coordinator (Retrieval + Risk + Generation)
├── retriever.py            # Hybrid Retrieval Engine (Dense + Sparse + Exact Boost)
├── embeddings.py           # SentenceTransformer neural embeddings & disk cache
├── generator.py            # Local Legal Synthesizer, LLM caller & Notice Drafter
├── ingestion.py            # PDF/TXT parsing & chunking for custom docs
├── requirements.txt        # Python dependencies
└── data/
    ├── legal_database.json # Structured legal knowledge base
    ├── legal_laws.txt      # Statutory provisions reference
    ├── court_cases.txt     # Landmark judgments & case laws
    └── .cache_embeddings.pkl # Instant disk cache for dense vectors
```
