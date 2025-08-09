import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from schemas import HackathonInput, HackathonOutput
from document_processor import get_documents_from_url
from vectorstore_utils import get_or_create_vectorstore
from rerank import rerank_with_bm25
from rag_chain import create_rag_chain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Please set OPENAI_API_KEY in your .env file.")

app = FastAPI(
    title="Advanced Document Q&A System (Modular)",
    description="An optimized API that answers questions based on documents from a URL."
)

@app.post("/hackrx/run", response_model=HackathonOutput)
async def run_submission(request: HackathonInput):
    start_time = time.time()
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    chunked_documents = get_documents_from_url(request.documents)
    if not chunked_documents:
        raise HTTPException(status_code=400, detail="Could not process the provided document URL.")

    vectorstore = get_or_create_vectorstore(request.documents, chunked_documents, embeddings_model)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 20})

    final_answers = [None] * len(request.questions)
    main_rag_chain = create_rag_chain(llm)

    def process_single_question(question: str, index: int):
        try:
            retrieved_docs = retriever.invoke(question)
            reranked_docs = rerank_with_bm25(question, retrieved_docs)
            response = main_rag_chain.invoke({
                "docs": reranked_docs,
                "question": question
            })
            return index, response.strip()
        except Exception:
            return index, "Unable to determine from provided context."

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_idx = {
            executor.submit(process_single_question, question, i): i
            for i, question in enumerate(request.questions)
        }
        for future in as_completed(future_to_idx):
            idx, answer = future.result()
            final_answers[idx] = answer

    total_time = time.time() - start_time
    print(f"API request processed in {total_time:.2f} seconds.")
    return HackathonOutput(answers=final_answers)

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server...")
    print("API will be available at http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)