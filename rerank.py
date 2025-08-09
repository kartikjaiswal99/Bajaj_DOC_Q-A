from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

def rerank_with_bm25(query: str, docs: list) -> list:
    if not docs:
        return []
    tokenized_corpus = [doc.page_content.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split()
    doc_scores = bm25.get_scores(tokenized_query)
    scored_docs = list(zip(doc_scores, docs))
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored_docs]