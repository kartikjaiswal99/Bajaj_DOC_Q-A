# ==============================================================================
#
#      HACKATHON SUBMISSION: ADVANCED DOCUMENT Q&A SYSTEM (OpenAI & Pinecone)
#
# This script implements the required FastAPI endpoint (/hackrx/run) to
# answer questions based on online documents. It uses a multi-step RAG
# pipeline with OpenAI models and a Pinecone vector database.
#
# This is the main entry point that imports from modular components.
# ==============================================================================

from app import app

# This allows the application to be run with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)