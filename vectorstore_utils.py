import os
import hashlib
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def get_or_create_vectorstore(documents_url, chunked_documents, embeddings_model):
    doc_id = hashlib.sha256(documents_url.encode()).hexdigest()
    vectorstore_folder = "vectorstore_cache"
    index_path = os.path.join(vectorstore_folder, f"{doc_id}.faiss")
    if os.path.exists(index_path):
        return FAISS.load_local(vectorstore_folder, embeddings_model, index_name=doc_id, allow_dangerous_deserialization=True)
    else:
        # --- Batching to avoid OpenAI token limit errors ---
        MAX_TOKENS = 200000  # Lower for extra safety
        def count_tokens(text):
            return len(text) // 4  # Rough estimate: 1 token ≈ 4 characters

        batches = []
        current_batch = []
        current_tokens = 0
        for doc in chunked_documents:
            doc_tokens = count_tokens(doc.page_content)
            # If a single doc is too large, split its content
            if doc_tokens > MAX_TOKENS:
                text = doc.page_content
                for i in range(0, len(text), MAX_TOKENS * 4):
                    part = text[i:i + MAX_TOKENS * 4]
                    part_doc = type(doc)(page_content=part, metadata=doc.metadata)
                    batches.append([part_doc])
                continue
            if current_tokens + doc_tokens > MAX_TOKENS and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            current_batch.append(doc)
            current_tokens += doc_tokens
        if current_batch:
            batches.append(current_batch)

        vectorstore = None
        for batch in batches:
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, embeddings_model)
            else:
                batch_vs = FAISS.from_documents(batch, embeddings_model)
                vectorstore.merge_from(batch_vs)
        vectorstore.save_local(vectorstore_folder, index_name=doc_id)
        return vectorstore