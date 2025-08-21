#HACKATHON SUBMISSION: ADVANCED DOCUMENT Q&A SYSTEM
# This is the main entry point that imports from modular components.

from app import app

# This allows the application to be run with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)