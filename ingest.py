import os
import re
import shutil
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


load_dotenv()


DOCUMENTS_PATH = Path("documents")
CHROMA_PATH = Path("chroma_db")

COLLECTION_NAME = "local_documents"


# =========================================================
# LOAD DOCUMENTS
# =========================================================

def load_documents():

    documents = []

    for filepath in DOCUMENTS_PATH.iterdir():

        if not filepath.is_file():
            continue

        suffix = filepath.suffix.lower()

        try:

            if suffix == ".pdf":

                loaded = PyPDFLoader(
                    str(filepath)
                ).load()

                documents.extend(loaded)

            elif suffix == ".txt":

                loaded = TextLoader(
                    str(filepath),
                    encoding="utf-8"
                ).load()

                documents.extend(loaded)

            elif suffix == ".docx":

                loaded = Docx2txtLoader(
                    str(filepath)
                ).load()

                documents.extend(loaded)

        except Exception as exc:

            print(
                f"Could not read {filepath.name}: {exc}"
            )

    return documents


# =========================================================
# SECTION-AWARE CHUNKING
# =========================================================

def create_section_documents(documents):

    section_documents = []

    # Matches headings such as:
    #
    # 1. PROJECT OVERVIEW
    # 2. SUPPORTED DOCUMENTS
    # 3. RAG PIPELINE
    # 4. TECHNOLOGY STACK
    # 5. REQUIREMENTS
    #
    heading_pattern = re.compile(
        r"^\s*(\d+)\.\s+(.+?)\s*$",
        re.MULTILINE
    )

    for document in documents:

        text = document.page_content

        if not text.strip():
            continue

        matches = list(
            heading_pattern.finditer(text)
        )

        # If no numbered sections are found,
        # keep the original document.
        if not matches:

            section_documents.append(
                Document(
                    page_content=text,
                    metadata=document.metadata.copy()
                )
            )

            continue

        for index, match in enumerate(matches):

            section_number = match.group(1)
            section_title = match.group(2).strip()

            start = match.start()

            if index + 1 < len(matches):

                end = matches[index + 1].start()

            else:

                end = len(text)

            section_text = text[start:end].strip()

            if not section_text:
                continue

            metadata = document.metadata.copy()

            metadata["section_number"] = section_number
            metadata["section_title"] = section_title

            section_documents.append(
                Document(
                    page_content=section_text,
                    metadata=metadata
                )
            )

    return section_documents


# =========================================================
# SPLIT LARGE SECTIONS
# =========================================================

def create_chunks(section_documents):

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=800,

        chunk_overlap=150,

        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ],
    )

    chunks = splitter.split_documents(
        section_documents
    )

    return chunks


# =========================================================
# MAIN
# =========================================================

def main():

    if not os.getenv("GOOGLE_API_KEY"):

        raise RuntimeError(
            "GOOGLE_API_KEY is missing from the .env file."
        )

    if not DOCUMENTS_PATH.exists():

        DOCUMENTS_PATH.mkdir()

    documents = load_documents()

    if not documents:

        print(
            "No PDF, DOCX, or TXT files found "
            "in the documents folder."
        )

        return

    print(
        f"Loaded {len(documents)} pages/sections."
    )

    # -----------------------------------------------------
    # Create section-aware documents
    # -----------------------------------------------------

    sections = create_section_documents(
        documents
    )

    print(
        f"Detected {len(sections)} document sections."
    )

    # -----------------------------------------------------
    # Create smaller chunks
    # -----------------------------------------------------

    chunks = create_chunks(
        sections
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    # -----------------------------------------------------
    # Show detected sections
    # -----------------------------------------------------

    print()
    print("Detected sections:")

    seen_sections = set()

    for chunk in chunks:

        section_number = chunk.metadata.get(
            "section_number"
        )

        section_title = chunk.metadata.get(
            "section_title"
        )

        if section_number and section_title:

            key = (
                section_number,
                section_title
            )

            if key not in seen_sections:

                print(
                    f"  {section_number}. "
                    f"{section_title}"
                )

                seen_sections.add(key)

    # -----------------------------------------------------
    # Create embeddings
    # -----------------------------------------------------

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    # -----------------------------------------------------
    # Delete old Chroma database
    # -----------------------------------------------------

    if CHROMA_PATH.exists():

        print()
        print("Removing old ChromaDB...")

        shutil.rmtree(
            CHROMA_PATH
        )

    # -----------------------------------------------------
    # Create new Chroma database
    # -----------------------------------------------------

    Chroma.from_documents(

        documents=chunks,

        embedding=embeddings,

        persist_directory=str(
            CHROMA_PATH
        ),

        collection_name=COLLECTION_NAME,
    )

    print()
    print(
        f"Vector database created at:"
    )

    print(
        CHROMA_PATH.resolve()
    )

    print()
    print("Ingestion complete.")


if __name__ == "__main__":

    main()