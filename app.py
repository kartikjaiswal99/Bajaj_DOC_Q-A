# ==============================================================================
#
#           FASTAPI APPLICATION - MAIN SERVER
#
# This creates the API endpoint required by the hackathon.
# ==============================================================================

from typing import List
from functools import lru_cache
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, HTTPException, Header
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from openai import RateLimitError

# Import our custom modules
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME
from schemas import HackathonInput, HackathonOutput
from document_processor import build_knowledge_base_from_urls
from rag_chain import create_rag_chain

app = FastAPI(
    title="Advanced Document Q&A System",
    description="An API that answers questions based on a given set of documents."
)

# --- Initialize Global Objects for Efficiency ---
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

# --- Caching for Document Processing ---
# We wrap the function with lru_cache here to apply it within the app's lifecycle
cached_build_knowledge_base = lru_cache(maxsize=10)(build_knowledge_base_from_urls)

@app.post("/hackrx/run", response_model=HackathonOutput)
async def run_submission(request: HackathonInput, Authorization: str = Header(None)):
    """
    This endpoint processes documents from URLs and answers questions according
    to the hackathon specifications. It accepts a Bearer token but does not validate it.
    """
    # Step 1: Download, process, and chunk documents from the provided URL.
    chunked_documents = cached_build_knowledge_base(request.documents)
    if not chunked_documents:
        raise HTTPException(status_code=400, detail="Error: Could not process the provided document.")
    
    # Step 2: Setup Pinecone Vector Store for this request.
    # Create the index if it doesn't exist.
    if PINECONE_INDEX_NAME not in pinecone_client.list_indexes().names():
        print(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
        pinecone_client.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,  # Dimension for OpenAI text-embedding-3-small
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    
    # ULTRA-OPTIMIZED: Multi-threaded bulk processing with concurrent uploads
    
    print(f"🚀 Processing {len(chunked_documents)} documents with multi-threading...")
    start_time = time.time()
    
    # Extract text content and metadata
    texts = [doc.page_content for doc in chunked_documents]
    metadatas = [doc.metadata for doc in chunked_documents]
    
    # Step 1: Compute embeddings in smaller batches concurrently
    print("Computing embeddings in parallel batches...")
    embedding_batch_size = 50  # Smaller batches for parallel processing
    embeddings = []
    
    def compute_embedding_batch(batch_texts, batch_idx):
        """Compute embeddings for a batch with retry logic"""
        max_retries = 3
        for retry in range(max_retries):
            try:
                batch_embeddings = embeddings_model.embed_documents(batch_texts)
                print(f"  Embedding batch {batch_idx + 1} - {len(batch_texts)} docs")
                return batch_embeddings
            except RateLimitError:
                wait_time = (2 ** retry) + (batch_idx * 0.5)  # Staggered backoff
                print(f"  Rate limit in embedding batch {batch_idx + 1}, waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
            except Exception as e:
                print(f"  Error in embedding batch {batch_idx + 1}: {e}")
                if retry == max_retries - 1:
                    raise
                time.sleep(1)
        return None
    
    # Create embedding batches
    embedding_batches = [texts[i:i + embedding_batch_size] 
                        for i in range(0, len(texts), embedding_batch_size)]
    
    # Process embeddings concurrently
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_batch = {
            executor.submit(compute_embedding_batch, batch, i): i 
            for i, batch in enumerate(embedding_batches)
        }
        
        batch_results = [None] * len(embedding_batches)
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                batch_embeddings = future.result()
                if batch_embeddings:
                    batch_results[batch_idx] = batch_embeddings
            except Exception as e:
                print(f"  Failed embedding batch {batch_idx + 1}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to compute embeddings for batch {batch_idx + 1}")
    
    # Flatten embeddings while preserving order
    embeddings = []
    for batch_result in batch_results:
        if batch_result:
            embeddings.extend(batch_result)
    
    embed_time = time.time()
    print(f"  ✅ Computed {len(embeddings)} embeddings in {embed_time - start_time:.2f}s")
    
    if len(embeddings) != len(texts):
        raise HTTPException(status_code=500, detail="Embedding count mismatch")
    
    # Step 2: Upload to Pinecone with concurrent batches
    print("Uploading to Pinecone with concurrent batches...")
    vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings_model)
    
    upload_batch_size = 50  # Smaller batches for concurrent upload
    total_batches = (len(texts) + upload_batch_size - 1) // upload_batch_size
    
    def upload_batch(batch_data, batch_idx):
        """Upload a batch to Pinecone with retry logic"""
        batch_texts, batch_embeddings, batch_metadatas = batch_data
        max_retries = 3
        
        for retry in range(max_retries):
            try:
                batch_start = time.time()
                vectorstore.add_texts(
                    texts=batch_texts,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas
                )
                batch_end = time.time()
                print(f"  Upload batch {batch_idx + 1}/{total_batches} - {len(batch_texts)} docs in {batch_end - batch_start:.2f}s")
                return True
            except RateLimitError:
                wait_time = (2 ** retry) + (batch_idx * 0.3)  # Staggered backoff
                print(f"  Rate limit in upload batch {batch_idx + 1}, waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
            except Exception as e:
                print(f"  Error in upload batch {batch_idx + 1}: {e}")
                if retry == max_retries - 1:
                    raise
                time.sleep(1)
        return False
    
    # Create upload batches
    upload_batches = []
    for i in range(0, len(texts), upload_batch_size):
        batch_texts = texts[i:i + upload_batch_size]
        batch_embeddings = embeddings[i:i + upload_batch_size]
        batch_metadatas = metadatas[i:i + upload_batch_size]
        upload_batches.append((batch_texts, batch_embeddings, batch_metadatas))
    
    # Upload concurrently with limited workers to avoid overwhelming Pinecone
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_batch = {
            executor.submit(upload_batch, batch_data, i): i 
            for i, batch_data in enumerate(upload_batches)
        }
        
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                success = future.result()
                if not success:
                    raise HTTPException(status_code=500, detail=f"Failed to upload batch {batch_idx + 1}")
            except Exception as e:
                print(f"  Upload failed for batch {batch_idx + 1}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to upload batch {batch_idx + 1}")
    
    total_time = time.time()
    print(f"  TOTAL TIME: {total_time - start_time:.2f}s for {len(chunked_documents)} documents")
    print(f"  Performance: {len(chunked_documents) / (total_time - start_time):.1f} docs/second")
    
    retriever = vectorstore.as_retriever(
        search_type="similarity", 
        search_kwargs={
            "k": 20,  # Retrieve more documents for better coverage
            "fetch_k": 50  # Fetch more candidates before filtering
        }
    )    # Step 3: Create the main RAG chain.
    main_chain = create_rag_chain(retriever)

    # Step 4: Process questions concurrently for maximum speed
    print(f"🤖 Processing {len(request.questions)} questions concurrently...")
    question_start_time = time.time()
    
    def process_single_question(question, question_idx):
        """Process a single question with error handling"""
        try:
            print(f"  Processing Q{question_idx + 1}: '{question[:50]}{'...' if len(question) > 50 else ''}'")
            start = time.time()
            response_dict = main_chain.invoke({"question": question})
            end = time.time()
            answer = response_dict.get("answer", "Could not generate an answer.")
            print(f"  Q{question_idx + 1} answered in {end - start:.2f}s")
            return answer
        except Exception as e:
            print(f"  Error processing Q{question_idx + 1}: {e}")
            return "An error occurred while processing this question."
    
    # Process questions concurrently
    final_answers = [None] * len(request.questions)
    with ThreadPoolExecutor(max_workers=5) as executor:  # Process up to 5 questions simultaneously
        future_to_idx = {
            executor.submit(process_single_question, question, i): i 
            for i, question in enumerate(request.questions)
        }
        
        for future in as_completed(future_to_idx):
            question_idx = future_to_idx[future]
            try:
                answer = future.result()
                final_answers[question_idx] = answer
            except Exception as e:
                print(f"  Failed to process question {question_idx + 1}: {e}")
                final_answers[question_idx] = "An error occurred while processing this question."
    
    question_end_time = time.time()
    print(f"  ALL QUESTIONS PROCESSED in {question_end_time - question_start_time:.2f}s")
    print(f"  Average per question: {(question_end_time - question_start_time) / len(request.questions):.2f}s")

    return HackathonOutput(answers=final_answers)