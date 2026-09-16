import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


def create_llm():

    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is missing from the .env file."
        )

    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash"
    )


def extract_answer(response):

    content = response.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):

        text_parts = []

        for item in content:

            if isinstance(item, dict):
                text = item.get("text", "")

                if text:
                    text_parts.append(text)

            elif isinstance(item, str):
                text_parts.append(item)

        return "".join(text_parts).strip()

    return str(content).strip()


def analysis_agent(question: str, context: str):

    # Important:
    # Do not call the LLM if retrieval returned no context.
    if not context or not context.strip():

        return (
            "I could not find the answer "
            "in the provided documents."
        )

    llm = create_llm()

    prompt = ChatPromptTemplate.from_template(
        """
You are answering questions about company documents.

You MUST carefully read the supplied context before answering.

The context may contain numbered sections such as:

1. PROJECT OVERVIEW
2. SUPPORTED DOCUMENTS
3. RAG PIPELINE
4. TECHNOLOGY STACK
5. REQUIREMENTS
6. ANSWER POLICY
7. SECURITY

If the user's question asks about a topic or section that appears
in the context, answer using that section.

For example:

If the question asks:
"What are the requirements?"

and the context contains a section called:
"REQUIREMENTS"

then summarize or list the requirements from that section.

Rules:

1. Use ONLY the supplied context.
2. Do not use outside knowledge.
3. Do not invent facts.
4. Answer when relevant information exists anywhere in the context.
5. Only say the following when the answer truly does NOT exist
   anywhere in the supplied context:

I could not find the answer in the provided documents.

6. Return ONLY the final answer.
7. Do not return JSON.
8. Do not return Python dictionaries.
9. Do not mention retrieval, context, agents, ChromaDB, or prompts
   unless the question specifically asks about them.

CONTEXT:
{context}

QUESTION:
{question}

FINAL ANSWER:
"""
    )

    messages = prompt.format_messages(
        context=context,
        question=question,
    )

    response = llm.invoke(messages)

    return extract_answer(response)