import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


load_dotenv()


CHROMA_PATH = Path("chroma_db")
COLLECTION_NAME = "local_documents"


def main():

    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is missing from .env"
        )

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    vectorstore = Chroma(
        persist_directory=str(CHROMA_PATH),
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    data = vectorstore.get(
        include=[
            "documents",
            "metadatas"
        ]
    )

    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])

    print()
    print("=" * 70)
    print("CHROMADB INSPECTION")
    print("=" * 70)

    print()
    print("Total chunks:", len(documents))

    print()
    print("STORED SECTIONS")
    print("-" * 70)

    sections_found = set()

    for metadata in metadatas:

        section_number = metadata.get(
            "section_number"
        )

        section_title = metadata.get(
            "section_title"
        )

        if section_number and section_title:

            section = (
                f"{section_number}. "
                f"{section_title}"
            )

            sections_found.add(section)

    for section in sorted(sections_found):

        print(section)

    print()
    print("=" * 70)
    print("SEARCHING FOR 'REQUIREMENTS'")
    print("=" * 70)

    matches = 0

    for index, (document, metadata) in enumerate(
        zip(documents, metadatas)
    ):

        text = document.lower()

        title = str(
            metadata.get(
                "section_title",
                ""
            )
        ).lower()

        if (
            "requirements" in text
            or "requirements" in title
        ):

            matches += 1

            print()
            print("MATCH:", matches)
            print("-" * 70)

            print(
                "Section:",
                metadata.get(
                    "section_number",
                    ""
                ),
                metadata.get(
                    "section_title",
                    ""
                )
            )

            print()
            print(document)

    print()
    print("=" * 70)
    print(
        "Requirements matches:",
        matches
    )
    print("=" * 70)

    if matches == 0:

        print()
        print(
            "No 'REQUIREMENTS' section or text "
            "exists in ChromaDB."
        )

        print()
        print(
            "This means the problem is NOT retrieval."
        )

        print(
            "The source document does not contain "
            "the requested information."
        )

    else:

        print()
        print(
            "Requirements data exists in ChromaDB."
        )

        print(
            "The retrieval ranking needs to be improved."
        )


if __name__ == "__main__":
    main()