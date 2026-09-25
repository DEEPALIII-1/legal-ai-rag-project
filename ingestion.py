import os
import io

def parse_pdf(file_bytes_or_path):
    """
    Extracts text from a PDF file path or bytes using pypdf.
    """
    try:
        from pypdf import PdfReader
        if isinstance(file_bytes_or_path, (bytes, io.BytesIO)):
            reader = PdfReader(io.BytesIO(file_bytes_or_path) if isinstance(file_bytes_or_path, bytes) else file_bytes_or_path)
        else:
            reader = PdfReader(file_bytes_or_path)
            
        text = ""
        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Page {page_idx + 1} ---\n" + page_text
        return text
    except Exception as e:
        print(f"[Error] PDF parsing failed: {e}")
        return ""


def chunk_text(text, chunk_size=800, overlap=150):
    """
    Splits long legal text into overlapping chunks for indexing.
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        # Try to break at a newline or period if possible
        if end < text_len:
            last_break = text.rfind("\n", start, end)
            if last_break == -1 or last_break < start + (chunk_size // 2):
                last_break = text.rfind(". ", start, end)
            if last_break != -1 and last_break > start + (chunk_size // 2):
                end = last_break + 1
                
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
            
        start = end - overlap if end < text_len else text_len
        
    return chunks


def process_uploaded_document(file_name, file_bytes):
    """
    Processes an uploaded PDF or TXT file and returns a list of indexed legal document dicts.
    """
    if file_name.lower().endswith(".pdf"):
        full_text = parse_pdf(file_bytes)
    else:
        try:
            full_text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            full_text = file_bytes.decode("latin-1", errors="ignore")

    chunks = chunk_text(full_text)
    doc_entries = []

    for idx, ch in enumerate(chunks):
        doc_entries.append({
            "id": f"UPLOAD-{file_name}-CHUNK-{idx+1}",
            "title": f"Uploaded Doc: {file_name} (Section {idx+1})",
            "category": "User Uploaded Document",
            "act": f"Custom Document: {file_name}",
            "sections": f"Part {idx+1}",
            "summary": ch[:300] + ("..." if len(ch) > 300 else ""),
            "key_elements": [],
            "penalties_remedies": "Refer to terms within uploaded document",
            "landmark_cases": [],
            "practical_steps": [ch],
            "relevant_authorities": "Document Parties / Concerned Court",
            "risk_level": "Medium",
            "keywords": [file_name]
        })

    return doc_entries
