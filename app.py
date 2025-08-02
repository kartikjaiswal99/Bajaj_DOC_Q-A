# ==============================================================================
#
#           FASTAPI APPLICATION - MAIN SERVER
#
# This creates the API endpoint required by the hackathon.
# ==============================================================================

from typing import List
from functools import lru_cache
from fastapi import FastAPI, HTTPException, Header
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

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
    
    # # Upsert documents into Pinecone.
    # vectorstore = PineconeVectorStore.from_documents(
    #     chunked_documents,
    #     index_name=PINECONE_INDEX_NAME,
    #     embedding=embeddings_model
    # )
    
    # FIX: Use batching to add documents to Pinecone, avoiding OpenAI token limits.

    print("Upserting documents to Pinecone in batches...")

    vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings_model)

    batch_size = 100

    for i in range(0, len(chunked_documents), batch_size):

        batch = chunked_documents[i:i + batch_size]

        vectorstore.add_documents(batch)

        print(f"  - Upserted batch {i // batch_size + 1}")
        
    retriever = vectorstore.as_retriever()

    # Step 3: Create the main RAG chain.
    main_chain = create_rag_chain(retriever)

    # Step 4: Process each question and format the output.
    final_answers: List[str] = []
    for question in request.questions:
        try:
            print(f"Processing question: '{question}'")
            # Invoke the full pipeline to get our structured internal response.
            response_dict = main_chain.invoke({"question": question})
            
            # Extract the simple string answer required by the hackathon.
            final_answers.append(response_dict.get("answer", "Could not generate an answer."))

        except Exception as e:
            print(f"Error processing question '{question}': {e}")
            final_answers.append("An error occurred while processing this question.")

    return HackathonOutput(answers=final_answers)