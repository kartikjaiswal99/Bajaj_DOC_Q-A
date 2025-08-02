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
    def generate_sub_questions(parsed_query: dict) -> List[str]:
        questions: List[str] = []
        procedure = parsed_query.get("procedure")
        entities = parsed_query.get("entities",)

        if procedure:
            questions.append(f"What are the policy rules, coverage, and conditions for '{procedure}'?")

        for entity in entities:
            questions.append(f"What does the policy say about '{entity}' in relation to '{procedure if procedure else 'the claim'}'?")

        questions.append("Are there any general exclusions or waiting periods that might apply?")

        return list(set(questions))
    
    # Chain 3: Evidence Retrieval and Answering
    rag_prompt = ChatPromptTemplate.from_template(
        "You are a meticulous fact-checker. Answer the sub-question based ONLY on the provided context.\n"
        "Your goal is to extract the most precise and factual information available.\n"
        "- Prioritize direct quotes and specific details like numbers, time periods, percentages, and conditions.\n"
        "- If the information is not explicitly available in the context, you MUST state 'The information is not available in the provided context.'\n\n"
        "Context:\n{context}\n\n"
        "Sub-Question: {sub_question}"
    )


    def retrieve_and_format_context(sub_question: str) -> str:
        docs = retriever.invoke(sub_question, k=10)
        return "\n\n".join([doc.page_content for doc in docs])



    evidence_chain = (
        {"sub_question": RunnablePassthrough()}
        | RunnableLambda(lambda x: {"sub_question": x["sub_question"], "context": retrieve_and_format_context(x["sub_question"])})
        | rag_prompt
        | llm
        | StrOutputParser()
    )



    # Chain 4: Final Synthesis
    final_prompt = ChatPromptTemplate.from_template(
        "You are an expert AI assistant for an insurance company. Your task is to provide a clear, concise, and factual answer to the user's question based *only* on the provided evidence from the policy document.\n\n"
        "**CRITICAL INSTRUCTIONS:**\n"
        "1. Synthesize the evidence to form a single, direct answer.\n"
        "2. The answer must be a complete sentence or two, directly addressing the user's question.\n"
        "3. Do NOT add any conversational fluff, apologies, or introductory phrases like 'Based on the evidence...'.\n"
        "4. If the evidence is insufficient to form a definitive answer, state that the information is not specified in the document.\n"
        "5. Emulate the style of a professional policy expert providing a definitive clarification.\n\n"
        "**Original Question:** {question}\n\n"
        "--- Evidence Gathered ---\n{evidence}\n--- End of Evidence ---\n\n"
        "Based *only* on the evidence above, provide the final, direct answer.\n"
        "{format_instructions}"

    )

    final_parser = JsonOutputParser(pydantic_object=FinalAnswer)
    synthesis_chain = final_prompt.partial(format_instructions=final_parser.get_format_instructions()) | llm | final_parser

    # --- Full Orchestration Chain ---
    def orchestrate(input_dict: dict) -> dict:
        question = input_dict["question"]
        try:
            parsed = parsing_chain.invoke({"question": question})
            sub_questions = generate_sub_questions(parsed)  # Add missing question argument
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