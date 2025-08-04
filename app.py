# FASTAPI APPLICATION - MAIN SERVER

from typing import List
from functools import lru_cache
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, HTTPException, Header
from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import FAISS 

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
    
    # Create FAISS vectorstore with speed optimizations
    vectorstore = FAISS.from_documents(
        documents=chunked_documents,
        embedding=embeddings_model
    )
    
    # Create SPEED-OPTIMIZED retriever 
    retriever = vectorstore.as_retriever(
        search_type="similarity",  # Faster than MMR
        search_kwargs={
            "k": 8,               # Reduced for speed
            "fetch_k": 15         # Reduced pre-filter candidates
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
    with ThreadPoolExecutor(max_workers=4) as executor:  # Increased to 4 for speed
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