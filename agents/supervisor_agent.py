import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


# =========================================================
# CREATE LLM
# =========================================================

def create_llm():

    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. "
            "Add GOOGLE_API_KEY=your_key to your .env file."
        )

    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0,
    )


# =========================================================
# EXTRACT TEXT FROM GEMINI RESPONSE
# =========================================================

def extract_text(response):

    content = response.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):

        text_parts = []

        for item in content:

            if isinstance(item, dict):

                text = item.get("text")

                if text:
                    text_parts.append(text)

            elif isinstance(item, str):

                text_parts.append(item)

        return "".join(text_parts).strip()

    return str(content).strip()


# =========================================================
# SUPERVISOR AGENT
# =========================================================

def supervisor_agent(question: str):

    llm = create_llm()

    prompt = ChatPromptTemplate.from_template(
        """
You are the Supervisor Agent in a multi-agent AI system.

Your job is to determine which agent should handle
the user's question.

Available agents:

1. RAG
   - Used for questions about documents uploaded
     by the user.
   - Examples:
     "What does the document say?"
     "Summarize the document."
     "What are the requirements mentioned in the PDF?"

2. SQL
   - Used for questions about structured company
     database information.
   - Examples:
     "What were our total sales last month?"
     "Show the top 10 customers."
     "How many orders did we receive?"

3. WEB
   - Used for questions requiring current internet
     information.
   - Examples:
     "What is the latest Azure release?"
     "What happened in the news today?"

For this phase, only RAG is implemented.

Therefore:

- If the question is about uploaded/local documents,
  return exactly:

RAG

- If the question requires SQL, return exactly:

SQL

- If the question requires current web information,
  return exactly:

WEB

- If uncertain, return:

RAG

IMPORTANT:

Return ONLY one word.

Allowed outputs:

RAG
SQL
WEB

Do not explain your decision.

User Question:
{question}

Agent:
"""
    )

    messages = prompt.format_messages(
        question=question
    )

    response = llm.invoke(messages)

    result = extract_text(response).upper().strip()

    # -----------------------------------------------------
    # SAFETY CHECK
    # -----------------------------------------------------

    if result not in ["RAG", "SQL", "WEB"]:
        return "RAG"

    return result