import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


load_dotenv()

CHROMA_PATH = Path("chroma_db")
COLLECTION_NAME = "local_documents"


def create_vectorstore():

    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is missing from the .env file."
        )

    if not CHROMA_PATH.exists():
        raise RuntimeError(
            "chroma_db does not exist. Run ingest.py first."
        )

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    return Chroma(
        persist_directory=str(CHROMA_PATH),
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )


def extract_keywords(question: str):

    stop_words = {
        "what", "are", "is", "the", "a", "an",
        "of", "in", "on", "to", "for", "from",
        "does", "do", "mentioned", "document",
        "documents", "please", "tell", "me",
        "about", "how", "why", "which", "can",
        "could", "all", "list", "give"
    }

    words = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        question.lower()
    )

    return [
        word
        for word in words
        if word not in stop_words and len(word) > 2
    ]


def score_document(document, keywords):

    text = document.page_content.lower()

    section_title = document.metadata.get(
        "section_title", ""
    ).lower()

    score = 0

    for keyword in keywords:

        # Highest priority: section heading
        if keyword == section_title:
            score += 100

        elif keyword in section_title:
            score += 50

        # Content match
        score += text.count(keyword) * 5

    return score


def retrieval_agent(question: str):

    vectorstore = create_vectorstore()

    keywords = extract_keywords(question)

    # Get a large set of semantic candidates
    results = vectorstore.similarity_search_with_score(
        question,
        k=20
    )

    documents = []

    for document, distance in results:

        keyword_relevance = score_document(
            document,
            keywords
        )

        # Chroma distance: lower is better
        semantic_score = 1 / (1 + float(distance))

        final_score = (
            keyword_relevance +
            semantic_score
        )

        documents.append({
            "document": document,
            "score": final_score
        })

    # Highest score first
    documents.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    # Keep best 6
    best_documents = [
        item["document"]
        for item in documents[:6]
    ]

    # Build context
    context_parts = []

    for document in best_documents:

        content = document.page_content.strip()

        if not content:
            continue

        section_number = document.metadata.get(
            "section_number"
        )

        section_title = document.metadata.get(
            "section_title"
        )

        if section_number and section_title:

            context_parts.append(
                f"SECTION {section_number}: "
                f"{section_title}\n\n"
                f"{content}"
            )

        else:

            context_parts.append(content)

    context = "\n\n" + (
        "\n\n" + ("=" * 60) + "\n\n"
    ).join(context_parts)

    return {
        "question": question,
        "documents": best_documents,
        "context": context,
    }