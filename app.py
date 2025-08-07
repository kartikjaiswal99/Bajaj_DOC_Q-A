# FASTAPI APPLICATION - MAIN SERVER

from typing import List
from functools import lru_cache
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, HTTPException, Header
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain_openai import ChatOpenAI

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

def estimate_tokens(text):
    """Estimate token count for text (conservative approximation: 3 chars per token)"""
    return max(1, len(text) // 3)

def create_vectorstore_with_batched_embeddings(chunked_documents, embeddings_model):
    """Create FAISS vectorstore with token-based batched embeddings to handle large documents"""
    
    MAX_TOKENS_PER_BATCH = 200_000  # Reduced to stay well below OpenAI's 300k limit
    batches = []
    current_batch = []
    current_tokens = 0
    
    for doc in chunked_documents:
        doc_tokens = estimate_tokens(doc.page_content)
        
        # If adding this document would exceed the token limit, start a new batch
        if current_tokens + doc_tokens > MAX_TOKENS_PER_BATCH and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        
        current_batch.append(doc)
        current_tokens += doc_tokens
    
    # Add the last batch
    if current_batch:
        batches.append(current_batch)
    
    if not batches:
        raise Exception("No documents to process")
    
    # Create the vectorstore from the first batch
    vectorstore = FAISS.from_documents(batches[0], embeddings_model)
    
    # Add remaining batches
    for i, batch in enumerate(batches[1:], 2):
        try:
            vectorstore.add_documents(batch)
        except Exception as e:
            # Try with smaller sub-batches if it's a token limit issue
            if "max_tokens_per_request" in str(e) and len(batch) > 10:
                sub_batch_size = len(batch) // 2
                for j in range(0, len(batch), sub_batch_size):
                    sub_batch = batch[j:j + sub_batch_size]
                    try:
                        vectorstore.add_documents(sub_batch)
                    except Exception as sub_e:
                        continue
            else:
                continue
    
    return vectorstore

def rerank_chunks(query, docs, llm=None, fast_mode=False):
    """
    Rerank retrieved docs using LLM for relevance to the query.
    Process in smaller batches to avoid rate limits.
    """
    if fast_mode or len(docs) <= 3:
        # Fast mode: skip reranking for small result sets or when speed is priority
        return docs[:6]  # Return top 6 without reranking
    
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    scored = []
    # Process reranking in smaller batches to avoid rate limits
    batch_size = 3  # Reduced batch size for faster processing
    
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i + batch_size]
        batch_scores = []
        
        for doc in batch_docs:
            section = doc.metadata.get("section_header", "")
            prompt = (
                f"Query: {query}\n"
                f"Section: {section}\n"
                f"Chunk: {doc.page_content[:500]}...\n"  # Truncate for speed
                "Score the relevance of this chunk to the query on a scale of 1 (not relevant) to 10 (very relevant). Only output the number."
            )
            try:
                score_str = llm.invoke(prompt).strip()
                score = int(''.join(filter(str.isdigit, score_str)) or '0')
            except Exception:
                score = 0
            batch_scores.append((score, doc))
        
        scored.extend(batch_scores)
        
        # Reduced delay between batches
        if i + batch_size < len(docs):
            time.sleep(0.05)  # Reduced from 0.1s
    
    scored.sort(reverse=True, key=lambda x: x[0])
    return [doc for score, doc in scored]


@app.post("/hackrx/run", response_model=HackathonOutput)
async def run_submission(request: HackathonInput):
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
    start_time = time.time()
    
    # Initialize embeddings model
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Handle large documents with batched processing
    try:
        vectorstore = create_vectorstore_with_batched_embeddings(chunked_documents, embeddings_model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

    # Create SPEED-OPTIMIZED retriever 
    retriever = vectorstore.as_retriever(
        search_type="similarity",  # Faster than MMR
        search_kwargs={
            "k": 8               # Reduced for speed, still enough for reranking
        }
    )
    
    embed_time = time.time()
    
    # Step 3: Create the main RAG chain
    main_chain = create_rag_chain(retriever)

    # Step 4: ULTRA-FAST question processing without timeout constraints
    question_start_time = time.time()
    
    llm_rerank = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    def process_single_question(question, question_idx):
        """Process a single question optimized for speed"""
        try:
            start = time.time()
            
            # Retrieve top-k docs
            docs = retriever.invoke(question)
            
            # Fast mode: skip reranking for speed
            fast_mode = len(request.questions) > 3 or len(chunked_documents) > 1000
            if fast_mode:
                top_docs = docs[:6]  # Use top 6 without reranking
            else:
                # Rerank only for small document/question sets
                reranked_docs = rerank_chunks(question, docs, llm=llm_rerank, fast_mode=False)
                top_docs = reranked_docs[:6]
            
            # Create a new retriever-like object for these docs
            class StaticRetriever:
                def invoke(self, q):
                    return top_docs
            
            # Use a custom RAG chain with these docs
            custom_chain = create_rag_chain(StaticRetriever())
            response_dict = custom_chain.invoke({"question": question})
            end = time.time()
            answer = response_dict.get("answer", "Unable to determine from provided context.")
            return answer
        except Exception as e:
            return "Unable to determine from provided context."
    
    # Process questions in parallel with optimized workers and NO TIMEOUT
    final_answers = [None] * len(request.questions)
    
    # Optimize workers for speed
    if len(chunked_documents) > 2000:
        max_workers = 3  # More workers for large docs
    elif len(request.questions) > 3:
        max_workers = 4  # More workers for many questions
    else:
        max_workers = 2  # Fewer workers for small sets
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
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
                final_answers[question_idx] = "Unable to determine from provided context."
    
    question_end_time = time.time()
    total_time = question_end_time - start_time

    return HackathonOutput(answers=final_answers)