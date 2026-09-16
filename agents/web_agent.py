import os

from dotenv import load_dotenv
from ddgs import DDGS

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


def search_web(question):

    results = []

    try:

        with DDGS() as ddgs:

            search_results = ddgs.text(
                question,
                region="us-en",
                safesearch="moderate",
                max_results=5,
            )

            for result in search_results:

                results.append(
                    {
                        "title": result.get(
                            "title",
                            ""
                        ),
                        "url": result.get(
                            "href",
                            ""
                        ),
                        "snippet": result.get(
                            "body",
                            ""
                        ),
                    }
                )

    except Exception as exc:

        print(
            f"Web search error: {exc}"
        )

    return results


def format_search_results(results):

    if not results:
        return "No web search results were found."

    formatted = []

    for index, result in enumerate(
        results,
        start=1
    ):

        formatted.append(
            f"""
Result {index}

Title:
{result["title"]}

URL:
{result["url"]}

Snippet:
{result["snippet"]}
"""
        )

    return "\n".join(formatted)


def analyze_web_results(
    question,
    search_results
):

    llm = create_llm()

    prompt = ChatPromptTemplate.from_template(
        """
You are a Web Research Agent.

Answer the user's question using ONLY the
web search results provided below.

User question:
{question}

Web search results:
{search_results}

Rules:

1. Use only the supplied search results.
2. Do not invent facts.
3. Do not use information that is not present
   in the search results.
4. Give a concise, useful answer.
5. If the results do not contain enough
   information, say:

I could not find enough information on the web.

6. Do not mention internal agents.
7. Do not mention prompts.
8. Do not return JSON.
9. Do not return Python dictionaries.
10. Return ONLY the final answer.

Final answer:
"""
    )

    messages = prompt.format_messages(
        question=question,
        search_results=search_results,
    )

    response = llm.invoke(messages)

    return extract_text(response)


def web_agent(question):

    results = search_web(question)

    if not results:

        return (
            "I could not find enough information "
            "on the web."
        )

    formatted_results = format_search_results(
        results
    )

    answer = analyze_web_results(
        question=question,
        search_results=formatted_results,
    )

    return answer