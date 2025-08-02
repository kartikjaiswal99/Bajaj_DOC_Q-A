# ==============================================================================
#
#           RAG CHAIN IMPLEMENTATION
#
# The advanced RAG pipeline for processing questions and generating answers
# ==============================================================================

from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from schemas import ParsedQuery, FinalAnswer

def create_rag_chain(retriever):
    """
    Creates the full, multi-step RAG chain using LangChain Expression Language (LCEL).
    """
    # --- Models ---
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # --- Chain 1: Query Parser ---
    parser_prompt = ChatPromptTemplate.from_template(
        "Parse the user's question into its main procedure and other key entities.\n"
        "{format_instructions}\n"
        "Question: {question}"
    )
    query_parser = JsonOutputParser(pydantic_object=ParsedQuery)
    parsing_chain = parser_prompt.partial(format_instructions=query_parser.get_format_instructions()) | llm | query_parser


    # Chain 2: Sub-Question Generator
    def generate_sub_questions(parsed_query: dict, original_question: str) -> List[str]:
        """Generate comprehensive sub-questions for better retrieval"""
        questions = [original_question]  # Always include the original question
        procedure = parsed_query.get("procedure")
        entities = parsed_query.get("entities", [])

        if procedure:
            questions.extend([
                f"What are the specific terms, conditions, and coverage details for {procedure}?",
                f"What are the exclusions, waiting periods, or limitations for {procedure}?",
                f"What are the eligibility criteria and requirements for {procedure}?",
                f"Are there any age restrictions, time limits, or special conditions for {procedure}?"
            ])

        for entity in entities:
            if entity and len(entity.strip()) > 2:  # Only meaningful entities
                questions.extend([
                    f"What does the policy specifically say about {entity}?",
                    f"How does {entity} affect coverage or benefits?",
                    f"Are there specific rules or conditions related to {entity}?"
                ])

        # Add general comprehensive questions for insurance policies
        questions.extend([
            "What are the grace periods, waiting periods, and renewal conditions?",
            "What are the general exclusions and limitations in this policy?",
            "What are the claim procedures and documentation requirements?",
            "What are the benefit limits, sub-limits, and maximum amounts covered?"
        ])

        return list(set(questions))  # Remove duplicates
    
    # Chain 3: Evidence Retrieval and Answering
    rag_prompt = ChatPromptTemplate.from_template(
        "You are a meticulous fact-checker. Answer the sub-question based ONLY on the provided context.\n"
        "Your goal is to extract the most precise and factual information available.\n"
        "- Prioritize direct quotes and specific details like numbers, time periods, percentages, and conditions.\n"
        "- If the information is not explicitly available in the context, you MUST state 'The information is not available in the provided context.'\n\n"
        "Context:\n{context}\n\n"
        "Sub-Question: {sub_question}"
    )


    # ACCURACY-OPTIMIZED: Enhanced retrieval with better context formatting
    def retrieve_and_format_context(sub_question: str) -> str:
        """Retrieve and format context with relevance scoring"""
        docs = retriever.invoke(sub_question, k=15)  # Increased from 10 to 15
        
        if not docs:
            return "No relevant information found in the document."
        
        formatted_chunks = []
        for i, doc in enumerate(docs):
            chunk_text = doc.page_content.strip()
            chunk_meta = doc.metadata
            
            # Add source and position context
            source_info = f"[Chunk {chunk_meta.get('chunk_id', i)}/{chunk_meta.get('total_chunks', 'N/A')}]"
            formatted_chunks.append(f"{source_info}\n{chunk_text}")
        
        return "\n\n---\n\n".join(formatted_chunks)

    # Chain 3: Enhanced Evidence Retrieval and Answering  
    rag_prompt = ChatPromptTemplate.from_template(
        "You are an expert insurance policy analyst. Extract the EXACT information that answers the sub-question.\n\n"
        "CRITICAL RULES:\n"
        "1. Quote specific numbers, percentages, time periods, and amounts EXACTLY as written\n"
        "2. Include ALL relevant conditions, exclusions, and requirements\n"
        "3. If multiple conditions exist, list them all\n"
        "4. If information is not found, state 'Information not found in provided context'\n"
        "5. Do NOT make assumptions or generalizations\n"
        "6. Use direct quotes from the policy when possible\n\n"
        "Context from Policy Document:\n{context}\n\n"
        "Sub-question: {sub_question}\n\n"
        "Extract and provide the precise answer with exact quotes and references:"
    )



    evidence_chain = (
        {"sub_question": RunnablePassthrough()}
        | RunnableLambda(lambda x: {"sub_question": x["sub_question"], "context": retrieve_and_format_context(x["sub_question"])})
        | rag_prompt
        | llm
        | StrOutputParser()
    )



    # Chain 4: ACCURACY-OPTIMIZED Final Synthesis
    final_prompt = ChatPromptTemplate.from_template(
        "You are an expert insurance policy analyst. Synthesize the evidence to provide a precise, complete answer.\n\n"
        "SYNTHESIS RULES:\n"
        "1. Extract and combine ALL relevant facts from the evidence\n"
        "2. Include specific numbers, time periods, conditions, and requirements\n"
        "3. If evidence contains conflicting information, mention both\n"
        "4. If insufficient evidence, state 'The policy document does not specify this information'\n"
        "5. Provide a direct, factual answer without conversational phrases\n"
        "6. Include exact quotes when they directly answer the question\n\n"
        "ORIGINAL QUESTION: {question}\n\n"
        "EVIDENCE FROM POLICY:\n{evidence}\n\n"
        "Synthesize the evidence into a complete, accurate answer:\n"
        "{format_instructions}"
    )

    final_parser = JsonOutputParser(pydantic_object=FinalAnswer)
    synthesis_chain = final_prompt.partial(format_instructions=final_parser.get_format_instructions()) | llm | final_parser

    # --- Full Orchestration Chain ---
    def orchestrate(input_dict: dict) -> dict:
        question = input_dict["question"]
        try:
            parsed = parsing_chain.invoke({"question": question})
            sub_questions = generate_sub_questions(parsed, question)  # Add missing question argument
            evidence_list = evidence_chain.batch(sub_questions)
            formatted_evidence = "\n\n".join(f"Evidence for '{q}':\n{a}" for q, a in zip(sub_questions, evidence_list))
            
            return synthesis_chain.invoke({
                "question": question,
                "evidence": formatted_evidence
            })
        except Exception as e:
            # Fallback to a simpler RAG if the structured chain fails
            print(f"Advanced chain failed: {e}. Falling back to simple RAG.")
            context = retrieve_and_format_context(question)
            simple_prompt = ChatPromptTemplate.from_template(
                "Answer the question based ONLY on the context.\nContext: {context}\nQuestion: {question}"
            )
            simple_chain = simple_prompt | llm | StrOutputParser()
            answer_text = simple_chain.invoke({"context": context, "question": question})
            return {"answer": answer_text}

    return RunnableLambda(orchestrate)