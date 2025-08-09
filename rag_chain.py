from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

def create_rag_chain(llm: ChatOpenAI):
    prompt = ChatPromptTemplate.from_template("""
        You are an expert Q&A assistant. Your task is to answer the user's question based *only* on the provided context.
        Follow these rules:
        1. Be direct and concise in your answer (1-3 sentences).
        2. If the answer is not present in the context, you MUST state: 'The answer could not be found in the provided document.'
        3. Do not use any information outside of the provided context.

        CONTEXT:
        {context}

        QUESTION: {question}

        FINAL ANSWER:""")

    def format_context(docs: list) -> str:
        if not docs:
            return "No relevant context found."
        context_parts = []
        for i, doc in enumerate(docs[:6]):
            content = doc.page_content.strip()
            header = doc.metadata.get("section_header", "")
            source_info = f"[{i+1}]"
            if header:
                source_info += f" Section: {header}"
            context_parts.append(f"{source_info}\n{content}")
        return "\n\n".join(context_parts)

    chain = (
        RunnableLambda(lambda inputs: {
            "context": format_context(inputs["docs"]),
            "question": inputs["question"]
        })
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain