import streamlit as st
from rag_pipeline import load_file, store_docs, search

laws = load_file("data/legal_laws.txt")
cases = load_file("data/court_cases.txt")

store_docs(laws + cases)

st.set_page_config(page_title="Legal AI Assistant", layout="centered")

st.title("⚖️ Legal AI Assistant (RAG System)")
st.write("Ask any legal question and get relevant answers")

query = st.text_input("Enter your legal question")


def risk_level(query):
    query = query.lower()

    high_risk_keywords = [
        "fraud", "scam", "theft", "robbery", "murder",
        "assault", "cyber crime", "cybercrime", "hack",
        "blackmail", "stalking", "cheating", "fraudulent"
    ]

    score = 0

    for word in high_risk_keywords:
        if word in query:
            score += 1

    if score >= 2:
        return " Very High Risk Case"
    elif score == 1:
        return " Moderate Risk Case"
    else:
        return " Low Risk Case"


# ---------------- MAIN OUTPUT SECTION ----------------
if query:
    st.subheader("Risk Level")
    st.write(risk_level(query))

    results = search(query)

    st.subheader("Relevant Legal Information")

    for r in results:
        st.write("", r)