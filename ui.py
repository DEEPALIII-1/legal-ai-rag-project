import streamlit as st
import os
import sys

# Ensure UTF-8 output
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rag_pipeline import query_rag, get_pipeline, add_document
from generator import draft_legal_notice

st.set_page_config(
    page_title="NyayaAI - Advanced Legal AI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .risk-badge-critical {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        display: inline-block;
        border: 1px solid #F87171;
    }
    .risk-badge-high {
        background-color: #FFEDD5;
        color: #9A3412;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        display: inline-block;
        border: 1px solid #FB923C;
    }
    .risk-badge-moderate {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        display: inline-block;
        border: 1px solid #60A5FA;
    }
    .risk-badge-standard {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        display: inline-block;
        border: 1px solid #34D399;
    }
    .source-card {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "pipeline" not in st.session_state:
    with st.spinner("Initializing Hybrid Retrieval Index & Pre-loading Neural Embeddings..."):
        st.session_state.pipeline = get_pipeline()

if "uploaded_files_count" not in st.session_state:
    st.session_state.uploaded_files_count = 0


# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/scales--v1.png", width=64)
    st.title("⚖️ Configuration")
    
    st.markdown("### 🤖 Intelligence Engine")
    engine_choice = st.selectbox(
        "Synthesis Engine",
        ["Built-in Smart Legal Synthesizer (Offline)", "Google Gemini API", "Groq API", "OpenAI API"]
    )

    api_key = None
    provider = "gemini"

    if engine_choice == "Google Gemini API":
        provider = "gemini"
        api_key = st.text_input("Gemini API Key", type="password", help="Enter Google Gemini API key")
    elif engine_choice == "Groq API":
        provider = "groq"
        api_key = st.text_input("Groq API Key", type="password", help="Enter Groq API key")
    elif engine_choice == "OpenAI API":
        provider = "openai"
        api_key = st.text_input("OpenAI API Key", type="password", help="Enter OpenAI API key")

    st.markdown("---")
    st.markdown("### 🎯 Search & Retrieval Filters")
    
    pipeline = st.session_state.pipeline
    all_categories = ["All"] + sorted(list(set(m.get("category", "") for m in pipeline.doc_metadata if m.get("category"))))
    selected_category = st.selectbox("Legal Domain Category", all_categories)

    top_k = st.slider("Max Sources Retrieved (Top-K)", min_value=1, max_value=5, value=3)

    st.markdown("---")
    st.markdown("### 📁 Upload Custom Legal Doc")
    st.caption("Upload agreements, FIR copies, or lease deeds (PDF / TXT) to query them with RAG.")
    uploaded_file = st.file_uploader("Upload Document", type=["pdf", "txt"])

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        added_count = add_document(uploaded_file.name, file_bytes)
        st.success(f"✅ Indexed '{uploaded_file.name}' ({added_count} chunks)!")
        st.session_state.uploaded_files_count += 1

    st.markdown("---")
    st.info("💡 **Knowledge Base Coverage:**\n- Bharatiya Nyaya Sanhita (BNS) 2024\n- Bharatiya Nagarik Suraksha Sanhita (BNSS)\n- IPC, CrPC, IT Act 2008\n- Consumer Protection Act 2019\n- Negotiable Instruments Act 1881 (Sec 138)\n- Supreme Court & High Court Judgments")


# --- MAIN INTERFACE TABS ---
tab_search, tab_notice, tab_directory = st.tabs([
    "🔍 Legal AI Search & Advisory",
    "📝 Legal Notice Drafter",
    "📞 Emergency Helplines & Directory"
])


# ==========================================
# TAB 1: LEGAL AI SEARCH & ADVISORY
# ==========================================
with tab_search:
    st.markdown('<div class="main-header">⚖️ NyayaAI: Advanced Legal Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Authoritative Indian Law Intelligence, Statutory Sections, Actionable Procedures & Precedents</div>', unsafe_allow_html=True)

    # Quick prompt chips
    st.markdown("**Quick Prompts:**")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    quick_query = None
    if col_p1.button("💳 Online UPI / Bank Scam"):
        quick_query = "Someone cheated me online on UPI and took 50000 rupees. How to recover money and report?"
    if col_p2.button("🚫 Cheque Bounce Notice (138)"):
        quick_query = "My cheque of 2 lakhs bounced due to insufficient funds. What is the 30-day legal notice process?"
    if col_p3.button("🏠 Landlord Withholding Deposit"):
        quick_query = "Landlord is refusing to refund my security deposit after vacating flat. What legal action can I take?"
    if col_p4.button("👮 Police Refusing FIR"):
        quick_query = "Police refused to register an FIR for my complaint. What can I do under CrPC / BNSS?"

    col_p5, col_p6, col_p7, col_p8 = st.columns(4)
    if col_p5.button("💼 Fired Without Notice & Salary"):
        quick_query = "Company terminated me without notice period and withheld my salary. Labor court remedy?"
    if col_p6.button("🚗 Hit & Run Accident Claim"):
        quick_query = "Car hit my motorcycle and ran away. How to claim compensation under MACT and solatium scheme?"
    if col_p7.button("🛡️ Domestic Violence Relief"):
        quick_query = "Can a woman claim protection order, residence, and maintenance under Domestic Violence Act?"
    if col_p8.button("📜 IPC 420 vs New BNS Law"):
        quick_query = "What is IPC 420 in new Bharatiya Nyaya Sanhita (BNS) law?"

    query_input = st.text_input(
        "Ask any legal question in plain English or legal terms:",
        value=quick_query or "",
        placeholder="e.g. Someone is blackmailing me with private photos online, what are my legal rights and steps?"
    )

    search_button = st.button("🔍 Get Legal Analysis", type="primary")

    if (search_button or quick_query) and query_input:
        with st.spinner("Analyzing statutory provisions, judicial precedents, and drafting action plan..."):
            cat_filter = None if selected_category == "All" else selected_category
            response = query_rag(
                query=query_input,
                category=cat_filter,
                top_k=top_k,
                api_key=api_key,
                provider=provider
            )

        risk = response["risk_info"]
        retrieved_docs = response["retrieved_docs"]

        # Risk Banner
        st.markdown("---")
        risk_class = "risk-badge-standard"
        if "CRITICAL" in risk["level"]:
            risk_class = "risk-badge-critical"
        elif "HIGH" in risk["level"]:
            risk_class = "risk-badge-high"
        elif "MODERATE" in risk["level"]:
            risk_class = "risk-badge-moderate"

        st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 15px; padding: 12px 18px; background-color: #F8FAFC; border-radius: 10px; border: 1px solid #E2E8F0;">
            <div>
                <span class="{risk_class}">{risk['badge']} ({risk['level']})</span>
                <span style="margin-left: 15px; color: #475569; font-weight: 600;">⏱️ Action Window: {risk['urgency']}</span>
            </div>
            <div style="font-size: 0.9rem; color: #64748B;">
                <b>Identified Urgency:</b> {risk['action']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Main Generated Answer
        st.markdown(response["answer"])

        # Display Sources / Documents Accordion
        st.markdown("---")
        st.subheader("📚 Referenced Statutory Authorities & Precedents")
        
        for idx, doc in enumerate(retrieved_docs, 1):
            score_pct = int(doc.get("score", 0.0) * 100)
            with st.expander(f"Source #{idx}: {doc.get('title')} (Relevance Match: {score_pct}%)"):
                st.markdown(f"**Governing Act:** `{doc.get('act')}`")
                st.markdown(f"**Statutory Sections:** `{doc.get('sections')}`")
                st.markdown(f"**Legal Summary:** {doc.get('summary')}")
                
                if doc.get("landmark_cases"):
                    st.markdown("**Key Judicial Precedents:**")
                    for c in doc.get("landmark_cases"):
                        st.markdown(f"- *{c}*")
                        
                if doc.get("penalties_remedies"):
                    st.markdown(f"**Penalties / Remedies:** {doc.get('penalties_remedies')}")
                    
                if doc.get("relevant_authorities"):
                    st.markdown(f"**Competent Authority:** {doc.get('relevant_authorities')}")


# ==========================================
# TAB 2: LEGAL NOTICE DRAFTER
# ==========================================
with tab_notice:
    st.subheader("📝 Automated Legal Demand Notice Generator")
    st.write("Generate standardized, court-ready formal legal notices under Indian statutory provisions.")

    notice_type = st.selectbox(
        "Select Notice Type",
        [
            "Dishonour of Cheque Demand Notice (Section 138 NI Act)",
            "Tenant Security Deposit Refund Demand Notice",
            "General Breach of Contract & Cease and Desist Notice"
        ]
    )

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        sender_name = st.text_input("Claimant Name (You / Your Client)", "Rahul Sharma")
        recipient_name = st.text_input("Addressee / Opposing Party Name", "ABC Enterprises / Ramesh Kumar")
    with col_s2:
        recipient_address = st.text_area("Opposing Party Address", "Plot No. 12, Industrial Area, Sector 5, New Delhi - 110001")

    details = {}
    if "Cheque" in notice_type:
        st.markdown("##### 💳 Cheque Details")
        c1, c2, c3 = st.columns(3)
        with c1:
            details["cheque_no"] = st.text_input("Cheque Number", "409218")
            details["cheque_date"] = st.text_input("Cheque Date (DD/MM/YYYY)", "15/01/2026")
        with c2:
            details["bank_name"] = st.text_input("Drawee Bank Name & Branch", "State Bank of India, Connaught Place")
            details["cheque_amount"] = st.text_input("Cheque Amount (in INR)", "1,50,000")
        with c3:
            details["memo_date"] = st.text_input("Bank Return Memo Date", "22/01/2026")
            details["reason"] = st.selectbox("Reason for Return", ["Funds Insufficient", "Account Closed", "Refer to Drawer", "Stop Payment"])
        n_type_code = "cheque_bounce"

    elif "Tenant" in notice_type:
        st.markdown("##### 🏠 Tenancy Details")
        c1, c2 = st.columns(2)
        with c1:
            details["deposit_amount"] = st.text_input("Security Deposit Amount (in INR)", "45,000")
            details["vacate_date"] = st.text_input("Date Premises Vacated", "31/12/2025")
        with c2:
            details["flat_address"] = st.text_input("Rented Property Address", "Flat 302, Green View Apartments, Indiranagar, Bengaluru")
        n_type_code = "tenant_deposit"

    else:
        n_type_code = "general"

    if st.button("📄 Generate Legal Notice Draft", type="primary"):
        notice_text = draft_legal_notice(n_type_code, sender_name, recipient_name, recipient_address, details)
        st.markdown("### 📋 Generated Legal Notice:")
        st.code(notice_text, language="markdown")
        st.download_button(
            "💾 Download Notice Draft (.txt)",
            data=notice_text,
            file_name=f"Legal_Notice_{n_type_code}.txt",
            mime="text/plain"
        )


# ==========================================
# TAB 3: EMERGENCY HELPLINES & DIRECTORY
# ==========================================
with tab_directory:
    st.subheader("📞 Official Indian Legal Helplines & Portals")
    st.write("Direct contact numbers and portals for immediate emergency relief and grievance filing.")

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.markdown("""
        #### 🚨 Emergency National Numbers
        - **National Emergency Number:** `112` (Police, Fire, Ambulance 24x7)
        - **National Cybercrime Reporting Helpline:** `1930` (Golden Hour financial freeze)
        - **Women Helpline (All India):** `1091` / `181`
        - **National Commission for Women (NCW):** `7827170170` (24x7 WhatsApp helpline)
        - **Childline Helpline:** `1098`
        - **National Consumer Disputes Helpline:** `1915`
        - **NALSA (Free Legal Services / Aid):** `15100`
        """)

    with col_d2:
        st.markdown("""
        #### 🌐 Official Government Legal Portals
        - **Cybercrime Reporting Portal:** [cybercrime.gov.in](https://cybercrime.gov.in)
        - **E-Daakhil (Consumer Commission Online):** [edaakhil.nic.in](https://edaakhil.nic.in)
        - **RTI Online Application Portal:** [rtionline.gov.in](https://rtionline.gov.in)
        - **RBI Sachet (Illegal Loan Apps / Fraudulent Entities):** [sachet.rbi.org.in](https://sachet.rbi.org.in)
        - **Ministry of Labor Grievance (Samadhan):** [samadhan.labour.gov.in](https://samadhan.labour.gov.in)
        - **eCourts Services Case Status:** [services.ecourts.gov.in](https://services.ecourts.gov.in)
        """)