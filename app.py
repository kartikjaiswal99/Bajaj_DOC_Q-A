# FASTAPI APPLICATION - MAIN SERVER

from typing import List
from functools import lru_cache
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, HTTPException, Header
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS 

import math
from openai import RateLimitError

from schemas import HackathonInput, HackathonOutput
from document_processor import build_knowledge_base_from_urls
from rag_chain import create_rag_chain

import os
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

app = FastAPI(
    title="Advanced Document Q&A System",
    description="An API that answers questions based on a given set of documents."
)

# --- Caching for Document Processing ---
cached_build_knowledge_base = lru_cache(maxsize=10)(build_knowledge_base_from_urls)

def create_vectorstore_with_batched_embeddings(chunked_documents, embeddings_model):
    """Create FAISS vectorstore with batched embeddings to handle large documents"""
    
    # Calculate safe batch size (aim for ~250k tokens per batch, with safety margin)
    # Assuming average 150 tokens per chunk
    max_chunks_per_batch = 1000  # Conservative estimate for safety
    
    if len(chunked_documents) <= max_chunks_per_batch:
        # Small document - process normally
        print(f"Processing {len(chunked_documents)} chunks in single batch")
        return FAISS.from_documents(chunked_documents, embeddings_model)
    
    # Large document - process in batches
    print(f" Large document detected: {len(chunked_documents)} chunks")
    print(f" Processing in batches of {max_chunks_per_batch} chunks...")
    
    # Process first batch to create initial vectorstore
    first_batch = chunked_documents[:max_chunks_per_batch]
    print(f"Batch 1: {len(first_batch)} chunks")
    vectorstore = FAISS.from_documents(first_batch, embeddings_model)
    
    # Process remaining batches and add to vectorstore
    num_batches = math.ceil(len(chunked_documents) / max_chunks_per_batch)
    
    for i in range(1, num_batches):
        start_idx = i * max_chunks_per_batch
        end_idx = min((i + 1) * max_chunks_per_batch, len(chunked_documents))
        batch = chunked_documents[start_idx:end_idx]
        print(f"Batch {i + 1}: {len(batch)} chunks")
        
        try:
            # Add documents to existing vectorstore
            vectorstore.add_documents(batch)
        except Exception as e:
            print(f" Error in batch {i + 1}: {e}")
            # Continue processing other batches
            continue
    
    print(f"Successfully processed {len(chunked_documents)} chunks in {num_batches} batches")
    return vectorstore


@app.post("/hackrx/run", response_model=HackathonOutput)
async def run_submission(request: HackathonInput, Authorization: str = Header(None)):
    """
    This endpoint processes documents from URLs and answers questions according
    to the hackathon specifications. It accepts a Bearer token but does not validate it.
    """
    # Initialize embeddings model here after environment variables are loaded
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Step 1: Download, process, and chunk documents from the provided URL.
    chunked_documents = cached_build_knowledge_base(request.documents)
    if not chunked_documents:
        raise HTTPException(status_code=400, detail="Error: Could not process the provided document.")
    
    # Step 2: ULTRA-FAST FAISS local vector database (optimized)
    print(f"Creating optimized FAISS vectorstore for {len(chunked_documents)} documents...")
    start_time = time.time()
    
    # Initialize embeddings model
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Handle large documents with batched processing
    try:
        vectorstore = create_vectorstore_with_batched_embeddings(chunked_documents, embeddings_model)
    except Exception as e:
        print(f"Error creating vectorstore: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


    # # Create FAISS vectorstore with speed optimizations
    # vectorstore = FAISS.from_documents(
    #     documents=chunked_documents,
    #     embedding=embeddings_model
    # )
    
    # Create SPEED-OPTIMIZED retriever 
    retriever = vectorstore.as_retriever(
        search_type="similarity",  # Faster than MMR
        search_kwargs={
            "k": 6               # Reduced for speed
            # "fetch_k": 15         # Reduced pre-filter candidates
        }
    )
    
    embed_time = time.time()
    print(f"FAISS vectorstore created in {embed_time - start_time:.2f}s")
    
    # Step 3: Create the main RAG chain
    main_chain = create_rag_chain(retriever)

    # Step 4: ULTRA-FAST question processing without timeout constraints
    print(f"Processing {len(request.questions)} questions with optimized speed...")
    question_start_time = time.time()
    
    def process_single_question(question, question_idx):
        """Process a single question optimized for speed"""
        try:
            start = time.time()
            response_dict = main_chain.invoke({"question": question})
            end = time.time()
            answer = response_dict.get("answer", "Unable to determine from provided context.")
            print(f"Q{question_idx + 1} answered in {end - start:.2f}s")
            return answer
        except Exception as e:
            print(f"Error processing Q{question_idx + 1}: {e}")
            return "Unable to determine from provided context."
    
    # Process questions in parallel with optimized workers and NO TIMEOUT
    final_answers = [None] * len(request.questions)
    max_workers = 2 if len(chunked_documents) > 1000 else 3  # Reduce workers for large docs
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # with ThreadPoolExecutor(max_workers=4) as executor:  # Increased to 4 for speed
        future_to_idx = {
            executor.submit(process_single_question, question, i): i 
            for i, question in enumerate(request.questions)
        }
        
        # Remove timeout constraint - let it complete naturally
        for future in as_completed(future_to_idx):
            question_idx = future_to_idx[future]
            try:
                answer = future.result()  # No timeout - let it complete
                final_answers[question_idx] = answer
            except Exception as e:
                print(f"Question {question_idx + 1} error: {e}")
                final_answers[question_idx] = "Unable to determine from provided context."
    
    question_end_time = time.time()
    total_time = question_end_time - start_time
    print(f" TOTAL TIME: {total_time:.2f}s for {len(chunked_documents)} docs + {len(request.questions)} questions")
    print(f" Average per question: {(question_end_time - question_start_time) / len(request.questions):.2f}s")

    return HackathonOutput(answers=final_answers)