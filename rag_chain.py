#RAG CHAIN IMPLEMENTATION

from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

def create_rag_chain(retriever):
    # Fast model with minimal temperature
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # Using mini for speed
    
    # ULTRA-FAST: Direct retrieval and formatting
    def get_context_fast(question: str) -> str:
        """Ultra-fast context retrieval with minimal processing"""
        try:
            docs = retriever.invoke(question)
            if not docs:
                return "No relevant information found."
            
            # Minimal formatting for maximum speed
            context_parts = []
            for i, doc in enumerate(docs[:10]):  # Top 10 docs only
                content = doc.page_content.strip()
                context_parts.append(f"[{i+1}] {content}")
            
            return "\n\n".join(context_parts)
        except Exception:
            return "Error retrieving context."
    
    # CONCISE but ACCURATE prompt - optimized for brief, focused answers
    fast_prompt = ChatPromptTemplate.from_template("""
        You are an expert insurance policy analyst. Provide concise, accurate answers based strictly on the policy content below.

        ANSWER REQUIREMENTS:
        - Be direct and concise (1-3 sentences maximum)
        - Include only the essential details: key time periods, amounts, main conditions
        - Focus on answering the specific question asked
        - Use exact figures and terminology from the policy
        - If information is not in the context, state "Not specified in policy"

        POLICY CONTENT:
        {context}

        QUESTION: {question}

        CONCISE ANSWER:""")
    
    # ULTRA-FAST: Direct chain without complex orchestration
    def fast_rag_logic(inputs):
        question = inputs["question"]
        try:
            # Direct retrieval and answer generation
            context = get_context_fast(question)
            
            # Fast synthesis
            chain = fast_prompt | llm | StrOutputParser()
            answer = chain.invoke({
                "context": context,
                "question": question
            })
            
            # Ensure answer is concise
            answer = answer.strip()
            if len(answer) > 300:  # Limit length
                # Take first sentence if too long
                sentences = answer.split('.')
                answer = sentences[0] + '.' if sentences else answer[:300]
            
            return {"answer": answer}
            
        except Exception as e:
            print(f"Fast RAG error: {e}")
            return {"answer": "Unable to determine from provided context."}
    
    return RunnableLambda(fast_rag_logic)