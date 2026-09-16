import os
import json

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
        model="gemini-3.6-flash",
        temperature=0,
    )


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


def validation_agent(
    question,
    answer,
    evidence
):

    llm = create_llm()

    prompt = ChatPromptTemplate.from_template(
        """
You are a Final Validation Agent in an
enterprise multi-agent AI system.

Your job is to verify whether the proposed
answer is supported by the supplied evidence.

User question:
{question}

Proposed answer:
{answer}

Evidence:
{evidence}

Validation rules:

1. Check whether the answer actually answers
   the user's question.

2. Check whether the answer is supported by
   the supplied evidence.

3. Do not use outside knowledge.

4. Detect hallucinated facts.

5. Detect unsupported numbers.

6. Detect claims that are not present in
   the evidence.

7. If the answer is fully supported, return:

VALID

8. If the answer is unsupported or incorrect,
   return:

INVALID

9. Return ONLY one word:

VALID
or
INVALID
"""
    )

    messages = prompt.format_messages(
        question=question,
        answer=answer,
        evidence=evidence
    )

    response = llm.invoke(messages)

    result = extract_text(
        response
    ).upper().strip()

    if "VALID" in result and "INVALID" not in result:
        return "VALID"

    return "INVALID"