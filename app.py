from rag_pipeline import load_file, store_docs, search

# ----------------------------
# Load datasets
# ----------------------------
laws = load_file("data/legal_laws.txt")
cases = load_file("data/court_cases.txt")

store_docs(laws + cases)

print("\n Legal AI Assistant Ready ")
print("Type 'exit' to stop the program\n")


# ----------------------------
# UNIQUE FEATURE: Risk Detector
# ----------------------------
def risk_level(query):
    high_risk_keywords = [
        "fraud", "theft", "murder", "assault",
        "cyber crime", "hack", "scam", "robbery"
    ]

    for word in high_risk_keywords:
        if word in query.lower():
            return " High Legal Risk Case"
    return " Low/Normal Legal Query"


# ----------------------------
# Main loop
# ----------------------------
while True:
    query = input("\nAsk Legal Question: ")

    if query.lower() == "exit":
        print("\nExiting Legal AI Assistant... ")
        break

    # Show risk level
    print("\n Risk Level:", risk_level(query))

    # Retrieve results using RAG
    results = search(query)

    # Display answer
    print("\n Answer (Based on Legal Documents):\n")

    for i, r in enumerate(results, 1):
        print(f"{i}. {r}\n")