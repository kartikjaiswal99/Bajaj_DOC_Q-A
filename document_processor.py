# ==============================================================================
#
#           DOCUMENT PROCESSING UTILITIES
#
# Functions for downloading and processing documents from URLs
# ==============================================================================

import email
from functools import lru_cache
import os
import docx
import requests
import tempfile
from typing import List, Tuple
import fitz

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def _extract_text_from_pdf(file_path: str) -> str:
    """Extracts text from a PDF file using PyMuPDF."""
    doc = fitz.open(file_path)
    text = ""

    for page in doc:
        text += page.get_text()

    return text



def _extract_text_from_docx(file_path: str) -> str:
    """Extracts text from a DOCX file, including tables."""
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
    """Extracts text from an EML file using Python's built-in email library."""
    with open(file_path, 'rb') as f:
        msg = email.message_from_bytes(f.read())

    

    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()

            if content_type == "text/plain":
                payload = part.get_payload(decode=True)
                body += payload.decode(errors='ignore')
    else:
        payload = msg.get_payload(decode=True)
        body = payload.decode(errors='ignore')

    subject = msg.get('subject', '')

    return f"Subject: {subject}\n\n{body}"



@lru_cache(maxsize=10)
def build_knowledge_base_from_urls(document_url: str) -> List:  # Changed from Tuple[str,...] to str
    """
    Downloads a document from URL, extracts text using appropriate parser, and splits
    it into chunks. Supports PDF, Word documents, and email formats.
    """
    print(f"Building knowledge base for: {document_url}")
    docs_with_metadata: List = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            response = requests.get(document_url)  # Changed from url to document_url
            response.raise_for_status()
            
            # file_name = os.path.basename(url.split('?'))
            file_name = os.path.basename(document_url.split('?')[0])  # Changed from url to document_url
            if not file_name:
                if '.docx' in document_url.lower():  # Changed from url to document_url
                    file_name = "document.docx"
                elif '.eml' in document_url.lower():  # Changed from url to document_url
                    file_name = "document.eml"
                else:
                    file_name = "document.pdf"
            
            temp_path = os.path.join(temp_dir, file_name)  # Fixed indentation and moved outside if block
            
            with open(temp_path, 'wb') as f:
                f.write(response.content)
            
            print(f"  - Processing {file_name}...")
            text_content = ""
            
            if file_name.lower().endswith('.pdf'):
                text_content = _extract_text_from_pdf(temp_path)
            elif file_name.lower().endswith('.docx'):
                text_content = _extract_text_from_docx(temp_path)
            elif file_name.lower().endswith('.eml'):
                text_content = _extract_text_from_eml(temp_path)
            else:
                print(f"  - Unsupported file type: {file_name}. Skipping.")
                return []  # Return empty list instead of None
            
            docs_with_metadata.append(Document(page_content=text_content, metadata={"source_url": document_url}))

        except requests.RequestException as e:
            print(f"  - Failed to download {document_url}. Error: {e}")
            return []  # Return empty list instead of continuing
        except Exception as e:
            print(f"  - Failed to process {document_url}. Error: {e}")
            return []  # Return empty list instead of continuing

    if not docs_with_metadata:
        return []  # Return empty list instead of None

    # ACCURACY-OPTIMIZED: Larger chunks with smart splitting for insurance documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Much larger for complete context
        chunk_overlap=300,  # Substantial overlap to preserve connections
        separators=[
            "\n\n",  # Paragraph breaks
            "\n",    # Line breaks
            ". ",    # Sentence endings
            "; ",    # Clause separators
            ", ",    # Sub-clause separators
            " "      # Word boundaries
        ],
        keep_separator=True  # Preserve separators for better context
    )
    chunked_docs = text_splitter.split_documents(docs_with_metadata)
    
    # Add document structure metadata for better retrieval
    for i, doc in enumerate(chunked_docs):
        doc.metadata.update({
            "chunk_id": i,
            "total_chunks": len(chunked_docs),
            "file_type": file_name.split('.')[-1].lower() if '.' in file_name else 'unknown'
        })
    
    print(f"Split documents into {len(chunked_docs)} chunks with enhanced metadata.")
    return chunked_docs