import os
import json
import requests

def assess_risk(query, retrieved_docs):
    """
    Evaluates risk severity and urgency based on query and retrieved legal documents.
    """
    query_lower = query.lower()
    
    critical_triggers = [
        "murder", "rape", "assault", "suicide", "threat to life", "kill",
        "private photos", "nude", "sextortion", "revenge porn", "beaten",
        "kidnap", "domestic violence", "extortion", "blackmail"
    ]
    high_triggers = [
        "fraud", "scam", "theft", "robbery", "stole", "cyber crime", "hack",
        "upi fraud", "fir refused", "arrest", "bail", "accident", "hit and run",
        "cheating", "fake loan app", "police"
    ]
    medium_triggers = [
        "cheque bounce", "section 138", "tenant", "eviction", "security deposit",
        "unpaid salary", "fired", "defamation", "contract breach", "divorce", "alimony"
    ]

    for trigger in critical_triggers:
        if trigger in query_lower:
            return {
                "level": "CRITICAL RISK",
                "color": "#FF2B2B",
                "badge": "🚨 CRITICAL",
                "urgency": "Immediate Emergency Action Required (0-24 Hours)",
                "action": "Contact emergency helplines (112 / 1091 / 1930) or reach the nearest police station immediately."
            }

    for trigger in high_triggers:
        if trigger in query_lower:
            return {
                "level": "HIGH RISK",
                "color": "#FF8C00",
                "badge": "⚠️ HIGH RISK",
                "urgency": "Urgent Action Required (Within 24-72 Hours)",
                "action": "Preserve all electronic/physical evidence, register formal FIR/cyber complaint, and contact legal counsel."
            }

    for trigger in medium_triggers:
        if trigger in query_lower:
            return {
                "level": "MODERATE RISK",
                "color": "#1E90FF",
                "badge": "ℹ️ MODERATE",
                "urgency": "Statutory Action Window (15 to 30 Days)",
                "action": "Issue formal statutory legal demand notice and initiate conciliation/legal proceedings."
            }

    # Fallback to doc metadata risk level
    if retrieved_docs:
        doc_risk = retrieved_docs[0].get("risk_level", "Medium").upper()
        if "CRITICAL" in doc_risk:
            return {"level": "CRITICAL RISK", "color": "#FF2B2B", "badge": "🚨 CRITICAL", "urgency": "Immediate Attention", "action": "Take prompt emergency/legal steps."}
        elif "HIGH" in doc_risk:
            return {"level": "HIGH RISK", "color": "#FF8C00", "badge": "⚠️ HIGH RISK", "urgency": "Urgent Statutory Action", "action": "Initiate complaint and preserve documentation."}
            
    return {
        "level": "STANDARD INQUIRY",
        "color": "#28A745",
        "badge": "📋 STANDARD",
        "urgency": "General Advisory / Regulatory Timeline",
        "action": "Follow standard dispute resolution or statutory application procedure."
    }


def call_llm_api(prompt, system_instruction, api_key=None, provider="gemini"):
    """
    Calls an external LLM API if key is provided (Gemini, Groq, or OpenAI).
    """
    gemini_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    openai_key = api_key or os.environ.get("OPENAI_API_KEY")
    groq_key = api_key or os.environ.get("GROQ_API_KEY")

    # 1. Gemini
    if provider == "gemini" and gemini_key:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": f"{system_instruction}\n\n{prompt}"}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 2048
                }
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            pass

    # 2. Groq
    if (provider == "groq" or groq_key) and groq_key:
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {groq_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except Exception:
            pass

    # 3. OpenAI
    if (provider == "openai" or openai_key) and openai_key:
        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {openai_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except Exception:
            pass

    return None


def local_legal_synthesizer(query, retrieved_docs, risk_info):
    """
    Built-in high-accuracy legal answer synthesis engine.
    Produces comprehensive, professional, structured legal advice from retrieved context.
    """
    if not retrieved_docs:
        return (
            "### ⚖️ Legal Assessment\n\n"
            "No specific matching statute or court precedent was found with high confidence in the current knowledge base. "
            "However, under Indian jurisprudence:\n\n"
            "- If this involves a civil grievance or monetary loss, consider sending a formal Legal Notice through an Advocate.\n"
            "- If this discloses a cognizable criminal offense, you have the right to register an FIR under Section 173 BNSS / Section 154 CrPC at your local police station.\n"
            "- For emergency guidance, dial **112** (National Emergency), **1930** (Cyber Fraud), or **1915** (Consumer Disputes)."
        )

    primary_doc = retrieved_docs[0]
    
    # Synthesize sections
    sections_list = []
    penalties_list = []
    cases_list = []
    steps_list = []
    authorities_list = []

    for d in retrieved_docs:
        if d.get("sections") and d.get("sections") not in sections_list:
            sections_list.append(f"**{d.get('act', 'Statute')}**: {d.get('sections')}")
        if d.get("penalties_remedies") and d.get("penalties_remedies") not in penalties_list:
            penalties_list.append(d.get("penalties_remedies"))
        for c in d.get("landmark_cases", []):
            if c not in cases_list:
                cases_list.append(c)
        for s in d.get("practical_steps", []):
            if s not in steps_list:
                steps_list.append(s)
        auth = d.get("relevant_authorities")
        if auth and auth not in authorities_list:
            authorities_list.append(auth)

    response_parts = []

    # Title & Executive Summary
    response_parts.append(f"### ⚖️ Legal Overview: {primary_doc.get('title')}")
    response_parts.append(f"**Category:** `{primary_doc.get('category')}` | **Risk Assessment:** {risk_info['badge']} (`{risk_info['level']}`)")
    response_parts.append(f"\n{primary_doc.get('summary')}\n")

    # Applicable Statutes & Sections
    response_parts.append("#### 📜 Applicable Statutes & Sections")
    for sec in sections_list[:3]:
        response_parts.append(f"- {sec}")
    
    if penalties_list:
        response_parts.append(f"\n**Penalties & Legal Remedies:**\n{penalties_list[0]}\n")

    # Key Elements
    elements = primary_doc.get("key_elements", [])
    if elements:
        response_parts.append("#### 🔍 Essential Legal Ingredients to Establish")
        for el in elements:
            response_parts.append(f"- {el}")
        response_parts.append("")

    # Step-by-Step Action Plan
    response_parts.append("#### 🛠️ Recommended Actionable Procedure")
    for i, step in enumerate(steps_list[:6], 1):
        response_parts.append(f"{i}. {step}")
    response_parts.append("")

    # Landmark Cases & Precedents
    if cases_list:
        response_parts.append("#### 🏛️ Key Judicial Precedents & Case Laws")
        for c in cases_list[:3]:
            response_parts.append(f"- *{c}*")
        response_parts.append("")

    # Authorities & Helplines
    response_parts.append("#### 🏢 Competent Authorities & Reporting Channels")
    for a in authorities_list[:2]:
        response_parts.append(f"- {a}")
    response_parts.append(f"- **Recommended Timeline:** {risk_info['urgency']}")

    # Disclaimer
    response_parts.append("\n> **Disclaimer:** *This analysis is generated based on statutory Indian laws, BNS/BNSS updates, and judicial precedents for informational purposes. For active court representations, filing vakalatnama, or tailored legal strategy, consult a certified advocate.*")

    return "\n".join(response_parts)


def generate_rag_response(query, retrieved_docs, api_key=None, provider="gemini"):
    """
    Main RAG generation coordinator.
    Uses external LLM (Gemini/OpenAI/Groq) if configured, otherwise uses built-in smart legal synthesizer.
    """
    risk_info = assess_risk(query, retrieved_docs)

    # Format context for LLM prompt
    context_blocks = []
    for i, doc in enumerate(retrieved_docs, 1):
        context_blocks.append(
            f"[Source {i}]: {doc.get('title')}\n"
            f"Act: {doc.get('act')}\n"
            f"Sections: {doc.get('sections')}\n"
            f"Summary: {doc.get('summary')}\n"
            f"Penalties/Remedies: {doc.get('penalties_remedies')}\n"
            f"Precedents: {'; '.join(doc.get('landmark_cases', []))}\n"
            f"Action Steps: {'; '.join(doc.get('practical_steps', []))}\n"
            f"Authorities: {doc.get('relevant_authorities')}"
        )
    context_str = "\n\n".join(context_blocks)

    system_instruction = (
        "You are an elite, highly qualified Senior Legal AI Assistant specializing in Indian Law, "
        "the new Bharatiya Nyaya Sanhita (BNS) 2023, Bharatiya Nagarik Suraksha Sanhita (BNSS) 2023, "
        "IPC, CrPC, IT Act, Consumer Protection Act, and Supreme Court of India precedents.\n"
        "Provide thorough, legally sound, structured, and actionable guidance directly addressing the user's issue.\n"
        "Ground your advice strictly on the provided retrieved context while synthesizing clear executive takeaways, "
        "exact statutory section citations, key precedents, chronological action steps, competent authorities, and risk level."
    )

    user_prompt = (
        f"USER LEGAL QUERY:\n{query}\n\n"
        f"RETRIEVED LEGAL KNOWLEDGE CONTEXT:\n{context_str}\n\n"
        f"ASSESSED RISK LEVEL:\n{risk_info['level']} ({risk_info['urgency']})\n\n"
        "Please provide a comprehensive, authoritative legal opinion and action plan with clear headings."
    )

    # Try external LLM first if API key is present
    if api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY"):
        llm_response = call_llm_api(user_prompt, system_instruction, api_key=api_key, provider=provider)
        if llm_response:
            return llm_response, risk_info

    # High-precision local synthesis
    synthesized_text = local_legal_synthesizer(query, retrieved_docs, risk_info)
    return synthesized_text, risk_info


def draft_legal_notice(notice_type, sender_name, recipient_name, recipient_address, details_dict):
    """
    Generates standardized formal legal notice drafts.
    """
    if notice_type == "cheque_bounce":
        cheque_no = details_dict.get("cheque_no", "[CHEQUE NUMBER]")
        cheque_date = details_dict.get("cheque_date", "[CHEQUE DATE]")
        bank_name = details_dict.get("bank_name", "[BANK NAME]")
        cheque_amount = details_dict.get("cheque_amount", "[AMOUNT]")
        memo_date = details_dict.get("memo_date", "[RETURN MEMO DATE]")
        reason = details_dict.get("reason", "Funds Insufficient")

        return f"""LEGAL DEMAND NOTICE UNDER SECTION 138 OF THE NEGOTIABLE INSTRUMENTS ACT, 1881

REGISTERED A.D. / SPEED POST

Date: [CURRENT DATE]

TO:
{recipient_name}
{recipient_address}

FROM:
[Advocate Name / On Behalf of {sender_name}]
[Advocate Address & Contact]

SUB: LEGAL DEMAND NOTICE UNDER SECTION 138 READ WITH SECTION 141 OF THE NEGOTIABLE INSTRUMENTS ACT, 1881 FOR DISHONOUR OF CHEQUE NO. {cheque_no} DATED {cheque_date} FOR AN AMOUNT OF RS. {cheque_amount}/-

Sir/Madam,

Under instructions from and on behalf of my client, {sender_name}, I do hereby serve upon you the following statutory legal demand notice:

1. That you, the Addressee, had issued Cheque bearing No. {cheque_no} dated {cheque_date} drawn on {bank_name} for an amount of Rs. {cheque_amount}/- (Rupees [Amount in Words] only) in discharge of a legally enforceable debt and liability owed to my client.

2. That my client presented the said Cheque for encashment through their bankers. However, to my client's shock, the said Cheque was returned dishonoured and unpaid with the Return Memo dated {memo_date} bearing remarks: "{reason}".

3. That you, despite being well aware of the insufficiency of funds in your bank account, deliberately and dishonestly issued the aforementioned cheque with fraudulent intent.

4. NOW THEREFORE, through this statutory notice, my client hereby calls upon you to pay the entire cheque amount of Rs. {cheque_amount}/- within EXACTLY FIFTEEN (15) DAYS from the date of receipt of this notice, failing which my client shall be constrained to initiate criminal proceedings against you under Section 138 of the Negotiable Instruments Act, 1881, as amended up to date, before the competent Court of Judicial Magistrate, holding you liable for imprisonment up to two years and fine up to twice the cheque amount, along with claim for interim compensation under Section 143A.

Copy retained in my office for record and further legal action.

Yours faithfully,

_______________________
ADVOCATE FOR THE PAYEE
"""
    elif notice_type == "tenant_deposit":
        deposit_amount = details_dict.get("deposit_amount", "[DEPOSIT AMOUNT]")
        vacate_date = details_dict.get("vacate_date", "[DATE OF VACATING]")
        flat_address = details_dict.get("flat_address", "[RENTED PROPERTY ADDRESS]")

        return f"""LEGAL DEMAND NOTICE FOR UNLAWFUL WITHHOLDING OF SECURITY DEPOSIT

REGISTERED A.D. / SPEED POST

Date: [CURRENT DATE]

TO:
{recipient_name} (Landlord/Owner)
{recipient_address}

FROM:
{sender_name} (Former Tenant)
[Tenant Address & Contact]

SUB: FORMAL NOTICE DEMANDING REFUND OF INTEREST-FREE SECURITY DEPOSIT OF RS. {deposit_amount}/- FOR PREMISES: {flat_address}

Sir/Madam,

I, {sender_name}, do hereby state and demand as follows:

1. That I was a lawful tenant in respect of premises situated at {flat_address} pursuant to Lease Agreement dated [Agreement Date], having paid an advance security deposit of Rs. {deposit_amount}/-.

2. That upon completion of notice period, I peacefully handed over vacant possession of the premises to you on {vacate_date} in pristine and undamaged condition.

3. That despite repeated reminders, emails, and phone calls, you have wrongfully, unlawfully, and maliciously failed to refund the security deposit of Rs. {deposit_amount}/-.

4. PLEASE TAKE NOTICE that you are hereby called upon to transfer the full security deposit sum of Rs. {deposit_amount}/- along with interest at 18% p.a. into my bank account within FIFTEEN (15) DAYS of receipt of this notice, failing which I shall initiate appropriate civil proceedings for summary recovery under Order 37 CPC and criminal complaint for criminal breach of trust under BNS Section 316 / IPC Section 406 at your sole cost and consequence.

Yours sincerely,

_______________________
{sender_name}
"""
    else:
        return f"""FORMAL LEGAL NOTICE / CEASE AND DESIST DEMAND

Date: [CURRENT DATE]

TO:
{recipient_name}
{recipient_address}

FROM:
{sender_name}

SUB: FORMAL LEGAL NOTICE REGARDING DEFICIENCY, BREACH OF OBLIGATION & LEGAL VIOLATIONS

Sir/Madam,

Under instructions from my client, {sender_name}, you are hereby notified of the following:

1. That you have committed acts causing financial loss, mental harassment, and breach of statutory obligations against my client.
2. You are hereby given notice to rectify the aforesaid violation, refund the outstanding sums, and cease the unlawful conduct within FIFTEEN (15) DAYS from receipt of this notice.
3. In default, my client shall initiate appropriate criminal and civil proceedings before the competent courts of law at your risk, cost, and consequences.

Yours faithfully,

_______________________
{sender_name}
"""
