import os
import re
import email
import docx
import fitz
import requests
import tempfile
from functools import lru_cache
from typing import List, Tuple
from langchain_core.documents import Document

def _extract_text_from_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def _extract_text_from_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    text = []
    for para in doc.paragraphs:
        text.append(para.text)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                text.append(cell.text)
    return "\n".join(text)

def _extract_text_from_eml(file_path: str) -> str:
    with open(file_path, 'rb') as f:
        msg = email.message_from_bytes(f.read())
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                body += payload.decode(errors='ignore')
    else:
        payload = msg.get_payload(decode=True)
        body = payload.decode(errors='ignore')
    subject = msg.get('subject', '')
    return f"Subject: {subject}\n\n{body}"

def structure_aware_chunking(text: str) -> List[Tuple[str, str]]:
    heading_regex = re.compile(r'(^[A-Z][A-Z0-9\-\s:]{3,}$|^\d+\. .+)', re.MULTILINE)
    matches = list(heading_regex.finditer(text))
    chunks = []
    if not matches:
        paras = text.split('\n\n')
        for para in paras:
            if len(para.strip()) > 100:
                chunks.append(("", para.strip()))
        return chunks

    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        section_header = match.group().strip()
        section_text = text[start:end].strip()
        paras = section_text.split('\n\n')
        for para in paras:
            if len(para.strip()) > 100:
                chunks.append((section_header, para.strip()))
    return chunks

@lru_cache(maxsize=10)
def get_documents_from_url(document_url: str) -> List[Document]:
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            response = requests.get(document_url)
            response.raise_for_status()
            file_name = os.path.basename(document_url.split('?')[0]) or "document.tmp"
            temp_path = os.path.join(temp_dir, file_name)
            with open(temp_path, 'wb') as f:
                f.write(response.content)

            text_content = ""
            if file_name.lower().endswith('.pdf'):
                text_content = _extract_text_from_pdf(temp_path)
            elif file_name.lower().endswith('.docx'):
                text_content = _extract_text_from_docx(temp_path)
            elif file_name.lower().endswith('.eml'):
                text_content = _extract_text_from_eml(temp_path)

            if not text_content:
                return []

            chunk_tuples = structure_aware_chunking(text_content)
            docs = []
            for i, (header, chunk) in enumerate(chunk_tuples):
                docs.append(Document(
                    page_content=chunk,
                    metadata={
                        "source_url": document_url,
                        "section_header": header,
                        "chunk_id": i
                    }
                ))
            return docs
        except Exception:
            return []