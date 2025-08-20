# ==============================================================================
#
#      HACKATHON SUBMISSION: ADVANCED DOCUMENT Q&A SYSTEM (OpenAI & FAISS)
#
# This script implements the required FastAPI endpoint (/hackrx/run) to
# answer questions based on online documents. It uses a multi-step RAG
# (Retrieval-Augmented Generation) pipeline with OpenAI models and FAISS
# vector database for high-performance document question answering.
#
# SYSTEM OVERVIEW:
# ================
# 1. Document Processing: Downloads and extracts text from PDF, DOCX, EML files
# 2. Intelligent Chunking: Structure-aware text splitting for optimal context
# 3. Vector Embeddings: OpenAI text-embedding-3-small for semantic search
# 4. Vector Database: FAISS for fast similarity search and retrieval
# 5. Question Processing: Parallel processing with ThreadPoolExecutor
# 6. Answer Generation: OpenAI GPT-4o-mini for contextual responses
# 7. Performance Optimization: Caching, batching, and adaptive processing
#
# FEATURES:
# =========
# - Multi-format document support (PDF, DOCX, EML)
# - Structure-aware document chunking with metadata preservation
# - High-performance FAISS vector search with similarity scoring
# - Parallel question processing for improved throughput
# - Intelligent document reranking for relevance optimization
# - Comprehensive error handling and graceful degradation
# - LRU caching for document processing efficiency
# - Token management for OpenAI API optimization
# - Adaptive processing modes based on dataset size
#
# USAGE:
# ======
# This is the main entry point that imports from modular components.
# Run the server with: python main_new.py
# Or with uvicorn: uvicorn main_new:app --reload
#
# API ENDPOINT:
# =============
# POST /hackrx/run
# - Input: Document URL and list of questions
# - Output: Corresponding answers based on document content
#
# ==============================================================================

from app import app

# Main application entry point
# This allows the application to be run with: uvicorn main_new:app --reload
if __name__ == "__main__":
    import uvicorn
    
    # Start the FastAPI server
    # Configuration can be adjusted via environment variables
    uvicorn.run(
        app, 
        host="0.0.0.0",  # Listen on all interfaces
        port=8000,       # Default port (configurable via APP_PORT env var)
        reload=False     # Set to True for development
    )