import sys
import os

# Ensure UTF-8 console output on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rag_pipeline import query_rag, get_pipeline, add_document
from generator import draft_legal_notice

def print_banner():
    banner = """
================================================================================
   ⚖️   ADVANCED LEGAL AI ASSISTANT (HYBRID RAG SYSTEM)
   Covering: BNS & BNSS 2024, IPC, CrPC, IT Act, Consumer Protection, NI Act
================================================================================
    """
    print(banner)

def print_help():
    print("""
Available Commands:
  - <your legal query> : Ask any question in plain English or legal terms
  - notice             : Launch Interactive Legal Demand Notice Generator
  - upload <file_path> : Ingest a custom PDF/TXT file into active RAG memory
  - categories         : List all supported legal domain categories
  - help               : Show this help menu
  - exit / quit        : Close the assistant
""")

def run_notice_wizard():
    print("\n--- 📝 Interactive Legal Demand Notice Generator ---")
    print("1. Cheque Dishonour Demand Notice (Section 138 NI Act)")
    print("2. Tenant Security Deposit Refund Notice")
    print("3. General Breach of Contract / Cease and Desist Notice")
    
    choice = input("\nSelect Notice Type [1/2/3]: ").strip()
    sender_name = input("Your Full Name (Claimant): ").strip() or "Client Name"
    recipient_name = input("Opposing Party / Addressee Name: ").strip() or "Recipient Name"
    recipient_address = input("Opposing Party Address: ").strip() or "[Full Address]"

    details = {}
    if choice == "1":
        details["cheque_no"] = input("Cheque Number: ").strip() or "123456"
        details["cheque_date"] = input("Cheque Date (DD/MM/YYYY): ").strip() or "01/01/2026"
        details["bank_name"] = input("Drawee Bank Name & Branch: ").strip() or "State Bank of India"
        details["cheque_amount"] = input("Cheque Amount (in INR): ").strip() or "50,000"
        details["memo_date"] = input("Return Memo Date (DD/MM/YYYY): ").strip() or "10/01/2026"
        details["reason"] = input("Dishonour Reason (e.g. Funds Insufficient): ").strip() or "Funds Insufficient"
        notice_type = "cheque_bounce"
    elif choice == "2":
        details["deposit_amount"] = input("Security Deposit Amount (in INR): ").strip() or "40,000"
        details["vacate_date"] = input("Date Premises Handed Over: ").strip() or "End of Tenancy"
        details["flat_address"] = input("Rented Property Address: ").strip() or "[Rented Property Address]"
        notice_type = "tenant_deposit"
    else:
        notice_type = "general"

    notice_draft = draft_legal_notice(notice_type, sender_name, recipient_name, recipient_address, details)
    print("\n" + "=" * 70)
    print("GENERATED LEGAL NOTICE DRAFT:")
    print("=" * 70)
    print(notice_draft)
    print("=" * 70)

def main():
    print_banner()
    print("Initializing Hybrid Retrieval Index & Pre-loading Neural Embeddings...")
    pipeline = get_pipeline()
    total_docs = len(pipeline.doc_metadata)
    print(f"✅ System Ready! Loaded {total_docs} authoritative legal topics & precedents.")
    print("Type your legal question below or type 'help' for options.\n")

    current_category = None

    while True:
        try:
            prompt_label = f"[{current_category}] Ask Legal Question > " if current_category else "Ask Legal Question > "
            query = input(prompt_label).strip()

            if not query:
                continue

            if query.lower() in ["exit", "quit", "q"]:
                print("\nExiting Advanced Legal AI Assistant. Goodbye!")
                break

            if query.lower() == "help":
                print_help()
                continue

            if query.lower() == "notice":
                run_notice_wizard()
                continue

            if query.lower() == "categories":
                cats = sorted(list(set(m.get("category", "") for m in pipeline.doc_metadata if m.get("category"))))
                print("\nSupported Legal Categories:")
                for i, c in enumerate(cats, 1):
                    print(f"  {i}. {c}")
                print("\nTip: Type 'filter <category name>' to restrict search, or 'filter all' to clear.")
                continue

            if query.lower().startswith("filter "):
                filter_val = query[7:].strip()
                if filter_val.lower() == "all":
                    current_category = None
                    print("Category filter cleared. Searching all legal domains.")
                else:
                    current_category = filter_val
                    print(f"Active filter set to: '{current_category}'")
                continue

            if query.lower().startswith("upload "):
                filepath = query[7:].strip()
                if not os.path.exists(filepath):
                    print(f"❌ File not found at: {filepath}")
                    continue
                try:
                    with open(filepath, "rb") as f:
                        file_bytes = f.read()
                    filename = os.path.basename(filepath)
                    added = add_document(filename, file_bytes)
                    print(f"✅ Successfully ingested '{filename}' ({added} chunks added to active index)!")
                except Exception as e:
                    print(f"❌ Error uploading file: {e}")
                continue

            # Execute RAG query
            print("\n🔍 Analyzing legal query through Hybrid Semantic & Statutory Engine...")
            response = query_rag(query, category=current_category, top_k=3)

            risk = response["risk_info"]
            print("\n" + "=" * 70)
            print(f"LEGAL RISK LEVEL: {risk['badge']} ({risk['level']})")
            print(f"ACTION WINDOW   : {risk['urgency']}")
            print("=" * 70 + "\n")

            print(response["answer"])

            # Display confidence scores & sources
            print("\n" + "-" * 70)
            print("RETRIEVED STATUTORY SOURCES & CONFIDENCE:")
            for i, doc in enumerate(response["retrieved_docs"], 1):
                title = doc.get("title", "Document")
                sec = doc.get("sections", "N/A")
                score = doc.get("score", 0.0)
                print(f"  [{i}] {title} | Sections: {sec} | Match: {score*100:.1f}%")
            print("-" * 70 + "\n")

        except KeyboardInterrupt:
            print("\nInterrupted. Exiting...")
            break
        except Exception as e:
            print(f"\n[Error processing query]: {e}\n")

if __name__ == "__main__":
    main()