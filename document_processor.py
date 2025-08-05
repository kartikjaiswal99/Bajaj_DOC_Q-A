#DOCUMENT PROCESSING UTILITIES

import email
from functools import lru_cache
import os
import docx
import requests
import tempfile
from typing import List, Tuple, Optional
import fitz
import re

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def _extract_text_from_pdf_batched(file_path: str, max_chars_per_batch: int = 500000) -> str:
    """Extracts text from a PDF file using PyMuPDF with batching for large documents."""
    doc = fitz.open(file_path)
    text = ""
    total_pages = len(doc)
    
    print(f"Processing PDF with {total_pages} pages")
    
    # Process pages in batches to avoid memory issues
    batch_size = max(1, total_pages // 10)  # Process in ~10 batches
    current_batch_text = ""
    
    for page_num in range(total_pages):
        page = doc[page_num]
        page_text = page.get_text()
        
        current_batch_text += page_text
        
        # If batch is getting too large, process it
        if len(current_batch_text) > max_chars_per_batch:
            text += current_batch_text
            current_batch_text = ""
            print(f"Processed batch up to page {page_num + 1}/{total_pages}")
    
    # Add remaining text
    if current_batch_text:
        text += current_batch_text
    
    doc.close()
    return text

def _extract_text_from_pdf(file_path: str) -> str:
    """Extracts text from a PDF file using PyMuPDF with batching for large documents."""
    try:
        return _extract_text_from_pdf_batched(file_path)
    except Exception as e:
        print(f"Batched extraction failed, trying standard method: {e}")
        # Fallback to original method
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
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



def structure_aware_chunking(text: str) -> List[Tuple[str, str]]:
    """
    Splits text into chunks based on section headings (structure-aware).
    Returns a list of (section_header, chunk) tuples.
    Optimized for very large documents and speed.
    """
    # Estimate document size
    text_length = len(text)
    print(f"Processing document of {text_length:,} characters")
    
    # Simplified regex for headings: lines in ALL CAPS or starting with numbers
    heading_regex = re.compile(r'(^[A-Z][A-Z0-9\-\s:]{3,}$|^\d+\. .+)', re.MULTILINE)
    matches = list(heading_regex.finditer(text))
    chunks = []
    
    if not matches:
        # Fallback: split by paragraphs with size limits
        paras = text.split('\n\n')
        current_chunk = ""
        current_size = 0
        max_chunk_size = 1500  # Reduced from 2000 for speed
        
        for para in paras:
            para = para.strip()
            if len(para) > 100:
                if current_size + len(para) > max_chunk_size and current_chunk:
                    chunks.append(("", current_chunk.strip()))
                    current_chunk = para
                    current_size = len(para)
                else:
                    current_chunk += "\n\n" + para if current_chunk else para
                    current_size += len(para)
        
        if current_chunk:
            chunks.append(("", current_chunk.strip()))
        return chunks
    
    # Process by sections
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        section_header = match.group().strip()
        section_text = text[start:end].strip()
        
        # For very large sections, split further
        if len(section_text) > 3000:  # Reduced from 5000
            # Split large sections by paragraphs
            paras = section_text.split('\n\n')
            current_chunk = ""
            current_size = 0
            max_chunk_size = 1500  # Reduced from 2000
            
            for para in paras:
                para = para.strip()
                if len(para) > 100:
                    if current_size + len(para) > max_chunk_size and current_chunk:
                        chunks.append((section_header, current_chunk.strip()))
                        current_chunk = para
                        current_size = len(para)
                    else:
                        current_chunk += "\n\n" + para if current_chunk else para
                        current_size += len(para)
            
            if current_chunk:
                chunks.append((section_header, current_chunk.strip()))
        else:
            # Small section - split by paragraphs
            paras = section_text.split('\n\n')
            for para in paras:
                if len(para.strip()) > 100:
                    chunks.append((section_header, para.strip()))
    
    print(f"Created {len(chunks)} chunks from {len(matches)} sections")
    return chunks


@lru_cache(maxsize=10)
def build_knowledge_base_from_urls(document_url: str) -> List:  
    """
    Downloads a document from URL, extracts text using appropriate parser, and splits
    it into chunks. Supports PDF, Word documents, and email formats.
    Handles very large documents with progressive processing.
    """
    print(f"Building knowledge base for: {document_url}")
    docs_with_metadata: List = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            response = requests.get(document_url) 
            response.raise_for_status()
            file_name = os.path.basename(document_url.split('?')[0])  
            if not file_name:
                if '.docx' in document_url.lower():  
                    file_name = "document.docx"
                elif '.eml' in document_url.lower(): 
                    file_name = "document.eml"
                else:
                    file_name = "document.pdf"
            temp_path = os.path.join(temp_dir, file_name)  
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
                return []  
            
            # Check if document is too large and process in chunks if needed
            if len(text_content) > 1000000:  # 1MB threshold
                print(f"Large document detected ({len(text_content):,} chars), processing in sections")
                docs_with_metadata = process_large_document(text_content, document_url, file_name)
            else:
                # Structure-aware chunking for normal documents
                chunk_tuples = structure_aware_chunking(text_content)
                for i, (section_header, chunk) in enumerate(chunk_tuples):
                    docs_with_metadata.append(Document(
                        page_content=chunk,
                        metadata={
                            "source_url": document_url,
                            "section_header": section_header,
                            "chunk_id": i,
                            "file_type": file_name.split('.')[-1].lower() if '.' in file_name else 'unknown'
                        }
                    ))
        except requests.RequestException as e:
            print(f"  - Failed to download {document_url}. Error: {e}")
            return []  
        except Exception as e:
            print(f"  - Failed to process {document_url}. Error: {e}")
            return [] 
    if not docs_with_metadata:
        return []  
    print(f"Split documents into {len(docs_with_metadata)} structure-aware chunks optimized for speed and context.")
    return docs_with_metadata

def process_large_document(text_content: str, document_url: str, file_name: str) -> List:
    """
    Process very large documents by splitting them into manageable sections
    before chunking to avoid token limit issues.
    """
    docs_with_metadata = []
    total_length = len(text_content)
    section_size = 500000  # 500KB sections
    section_count = 0
    
    print(f"Processing large document in {section_size:,} character sections")
    
    for i in range(0, total_length, section_size):
        section_text = text_content[i:i + section_size]
        section_count += 1
        
        print(f"Processing section {section_count} ({len(section_text):,} chars)")
        
        # Process this section with structure-aware chunking
        chunk_tuples = structure_aware_chunking(section_text)
        
        for j, (section_header, chunk) in enumerate(chunk_tuples):
            docs_with_metadata.append(Document(
                page_content=chunk,
                metadata={
                    "source_url": document_url,
                    "section_header": section_header,
                    "chunk_id": f"{section_count}_{j}",
                    "file_type": file_name.split('.')[-1].lower() if '.' in file_name else 'unknown',
                    "section_number": section_count
                }
            ))
    
    print(f"Processed {section_count} sections into {len(docs_with_metadata)} chunks")
    return docs_with_metadata