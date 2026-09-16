# 🤖 Multi-Agent AI Data & Knowledge Assistant

## Overview

The **Multi-Agent AI Data & Knowledge Assistant** is an AI-powered application that allows users to ask questions in natural language and receive answers from multiple data sources.

The application uses a **Supervisor Agent** to understand the user's question and route it to the appropriate specialized agent:

* **RAG Agent** → Searches local documents
* **SQL Agent** → Queries SQL Server data
* **Web Agent** → Searches the web for current information

The overall workflow is orchestrated using **LangGraph**, while **LangChain** provides the LLM, retrieval, prompt, and tool integration.

The goal of this project is to provide a single natural-language interface for accessing both **structured and unstructured data**.

---

## Features

### 🤖 Multi-Agent Architecture

* Supervisor Agent for intelligent question routing
* Specialized RAG, SQL, and Web agents
* LangGraph-based workflow orchestration
* Agent validation and retry workflow

### 📚 RAG — Retrieval-Augmented Generation

* Supports local PDF, DOCX, and TXT documents
* Document text extraction and chunking
* Gemini embeddings
* ChromaDB vector database
* Semantic document retrieval
* Context-based answer generation

### 🗄️ SQL Agent

* Natural-language database questions
* Automatic database schema inspection
* Natural-language-to-SQL generation
* SELECT-only SQL validation
* SQL Server execution using PyODBC
* Business-friendly result analysis
* SQL error handling and retry

### 🌐 Web Research Agent

* Web search using DuckDuckGo/DDGS
* Retrieves current information from the web
* Uses Gemini to summarize search results
* Provides concise natural-language answers

### 🛡️ Validation

* Generated SQL is validated before execution
* Agent responses can be validated against available evidence
* Retry handling for failed operations

---

## Architecture

```text
                         USER
                           │
                           ▼
                  ┌─────────────────┐
                  │   SUPERVISOR    │
                  │      AGENT      │
                  └────────┬────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
    ┌───────────┐    ┌───────────┐    ┌───────────┐
    │ RAG AGENT │    │ SQL AGENT │    │ WEB AGENT │
    └─────┬─────┘    └─────┬─────┘    └─────┬─────┘
          │                │                │
          ▼                ▼                ▼
     ┌─────────┐      ┌──────────┐      ┌──────────┐
     │ChromaDB │      │SQL Server│      │Web Search│
     └────┬────┘      └─────┬────┘      └────┬─────┘
          │                │                │
          └────────────────┼────────────────┘
                           ▼
                  ┌─────────────────┐
                  │ ANALYSIS AGENT  │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ VALIDATION      │
                  │     AGENT       │
                  └────────┬────────┘
                           │
                           ▼
                     FINAL ANSWER
```

---

## How It Works

### 1. User Question

The user enters a question in natural language.

Example:

```text
What technologies are used in the project?
```

The question is passed to the LangGraph workflow.

### 2. Supervisor Agent

The Supervisor Agent determines which specialized agent should process the question.

```text
Document question
        ↓
    RAG Agent

Database question
        ↓
    SQL Agent

Current information
        ↓
    Web Agent
```

### 3. RAG Agent

The RAG pipeline works as follows:

```text
User Question
      ↓
Gemini Embedding
      ↓
ChromaDB Vector Search
      ↓
Relevant Document Chunks
      ↓
Analysis Agent
      ↓
Final Answer
```

The system uses only the retrieved document context when generating document-based answers.

### 4. SQL Agent

The SQL workflow:

```text
Natural Language Question
          ↓
Database Schema
          ↓
Gemini SQL Generation
          ↓
SQL Validation
          ↓
SQL Server
          ↓
Query Results
          ↓
Business Analysis
          ↓
Final Answer
```

The SQL Agent is designed to allow read-oriented queries and blocks dangerous SQL operations such as:

```text
INSERT
UPDATE
DELETE
DROP
ALTER
CREATE
TRUNCATE
EXEC
```

### 5. Web Agent

The Web workflow:

```text
User Question
      ↓
DuckDuckGo Search
      ↓
Search Results
      ↓
Gemini
      ↓
Final Answer
```

This allows the application to answer questions requiring current web information.

### 6. Validation

After an agent produces an answer, the workflow can validate the response and retry when required.

---

## Project Structure

```text
multi-agent-ai-data-knowledge-assistant/
│
├── agents/
│   ├── __init__.py
│   ├── graph.py
│   ├── supervisor_agent.py
│   ├── retrieval_agent.py
│   ├── analysis_agent.py
│   ├── sql_agent.py
│   ├── web_agent.py
│   └── validation_agent.py
│
├── documents/
│   └── README.md
│
├── chat.py
├── ingest.py
├── test_retrieval.py
├── test_sql_agent.py
├── check_chroma.py
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

### File Responsibilities

| File                         | Responsibility                                      |
| ---------------------------- | --------------------------------------------------- |
| `chat.py`                    | Main application and user interaction               |
| `agents/graph.py`            | LangGraph workflow and orchestration                |
| `agents/supervisor_agent.py` | Routes questions to RAG, SQL, or Web                |
| `agents/retrieval_agent.py`  | Retrieves relevant document chunks                  |
| `agents/analysis_agent.py`   | Generates document-based answers                    |
| `agents/sql_agent.py`        | Generates, validates and executes SQL               |
| `agents/web_agent.py`        | Performs web research                               |
| `agents/validation_agent.py` | Validates generated answers                         |
| `ingest.py`                  | Processes documents and creates ChromaDB embeddings |
| `test_retrieval.py`          | Tests document retrieval                            |
| `test_sql_agent.py`          | Tests SQL functionality                             |
| `check_chroma.py`            | Inspects ChromaDB data                              |

---

## Technology Stack

| Technology    | Purpose                                   |
| ------------- | ----------------------------------------- |
| **Python**    | Main programming language                 |
| **LangChain** | LLM, prompts, retrieval and AI components |
| **LangGraph** |                                           |
