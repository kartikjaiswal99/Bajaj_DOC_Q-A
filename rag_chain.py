#RAG CHAIN IMPLEMENTATION

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

def create_rag_chain(retriever):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    def get_context_fast(question: str) -> str:
        """Ultra-fast context retrieval with minimal processing"""
        try:
            docs = retriever.invoke(question)
            if not docs:
                return "No relevant information found."
            
            # Limit to top 6 docs for speed
            context_parts = []
            for i, doc in enumerate(docs[:6]):  # Reduced from 10 to 6
                content = doc.page_content.strip()
                section = doc.metadata.get("section_header", "")
                if section:
                    context_parts.append(f"[{i+1}] Section: {section}\n{content[:800]}...")  # Truncate content
                else:
                    context_parts.append(f"[{i+1}] {content[:800]}...")  # Truncate content
            
            return "\n\n".join(context_parts)
        except Exception:
            return "Error retrieving context."
    
    # More concise prompt for faster processing
    fast_prompt = ChatPromptTemplate.from_template("""
        Answer the question based on the policy content below. Be direct and concise (1-2 sentences).

        POLICY CONTENT:
        {context}

        QUESTION: {question}

        ANSWER:""")
    
    def fast_rag_logic(inputs):
        question = inputs["question"]
        try:
            context = get_context_fast(question)
            chain = fast_prompt | llm | StrOutputParser()
            answer = chain.invoke({
                "context": context,
                "question": question
            })
            answer = answer.strip()
            if len(answer) > 200:  # Reduced from 300 to 200
                sentences = answer.split('.')
                answer = sentences[0] + '.' if sentences else answer[:200]
            return {"answer": answer}
        except Exception as e:
            return {"answer": "Unable to determine from provided context."}
    
    return RunnableLambda(fast_rag_logic)