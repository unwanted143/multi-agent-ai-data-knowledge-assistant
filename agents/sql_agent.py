import os
import re

import pyodbc

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


SERVER = os.getenv("SQL_SERVER", "SAI")
DATABASE = os.getenv("SQL_DATABASE", "Medtronic")
DRIVER = os.getenv(
    "SQL_DRIVER",
    "ODBC Driver 18 for SQL Server"
)

MAX_RETRIES = 3


def get_connection():

    connection_string = (
        f"DRIVER={{{DRIVER}}};"
        f"SERVER={SERVER};"
        f"DATABASE={DATABASE};"
        "Trusted_Connection=yes;"
        "Encrypt=yes;"
        "TrustServerCertificate=yes;"
    )

    return pyodbc.connect(
        connection_string
    )


def create_llm():

    if not os.getenv("GOOGLE_API_KEY"):

        raise RuntimeError(
            "GOOGLE_API_KEY is missing. "
            "Add it to your .env file."
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

        return "".join(
            text_parts
        ).strip()

    return str(content).strip()


def get_database_schema():

    connection = get_connection()

    cursor = connection.cursor()

    try:

        query = """
        SELECT
            TABLE_SCHEMA,
            TABLE_NAME,
            COLUMN_NAME,
            DATA_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS
        ORDER BY
            TABLE_SCHEMA,
            TABLE_NAME,
            ORDINAL_POSITION
        """

        cursor.execute(query)

        rows = cursor.fetchall()

        schema_lines = []

        for row in rows:

            schema_lines.append(
                f"{row.TABLE_SCHEMA}."
                f"{row.TABLE_NAME} "
                f"({row.COLUMN_NAME}: "
                f"{row.DATA_TYPE})"
            )

        return "\n".join(
            schema_lines
        )

    finally:

        cursor.close()
        connection.close()


def clean_sql(sql):

    sql = sql.replace(
        "```sql",
        ""
    )

    sql = sql.replace(
        "```",
        ""
    )

    return sql.strip()


def validate_sql(sql):

    sql_clean = sql.strip().lower()

    if not sql_clean:

        return False, "SQL query is empty."

    if not (
        sql_clean.startswith("select")
        or sql_clean.startswith("with")
    ):

        return (
            False,
            "Only SELECT or WITH queries are allowed."
        )

    forbidden_keywords = [

        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "create",
        "truncate",
        "merge",
        "exec",
        "execute",
        "grant",
        "revoke",

    ]

    for keyword in forbidden_keywords:

        pattern = rf"\b{keyword}\b"

        if re.search(
            pattern,
            sql_clean
        ):

            return (
                False,
                f"Forbidden SQL keyword detected: "
                f"{keyword}"
            )

    return True, ""


def generate_sql(
    question,
    schema,
    previous_sql=None,
    error=None
):

    llm = create_llm()

    if previous_sql and error:

        correction_context = f"""
Previous SQL:
{previous_sql}

Database error:
{error}

The previous SQL failed.

Generate a corrected SQL query.
"""

    else:

        correction_context = ""

    prompt = ChatPromptTemplate.from_template(
        """
You are an expert SQL Server Agent.

Convert the business user's question into
a READ-ONLY SQL Server query.

Database schema:

{schema}

User question:

{question}

{correction_context}

Rules:

1. Return ONLY SQL.
2. SQL must be SELECT or WITH.
3. Never modify database data.
4. Never use INSERT.
5. Never use UPDATE.
6. Never use DELETE.
7. Never use DROP.
8. Never use ALTER.
9. Never use CREATE.
10. Never use TRUNCATE.
11. Never use MERGE.
12. Never use EXEC.
13. Use only tables and columns from the schema.
14. Do not invent columns.
15. Do not invent tables.
16. Use SQL Server syntax.
17. Do not explain anything.
18. Do not return markdown.

SQL:
"""
    )

    messages = prompt.format_messages(
        question=question,
        schema=schema,
        correction_context=correction_context
    )

    response = llm.invoke(
        messages
    )

    return clean_sql(
        extract_text(response)
    )


def execute_sql(sql):

    valid, reason = validate_sql(
        sql
    )

    if not valid:

        raise ValueError(
            reason
        )

    connection = get_connection()

    cursor = connection.cursor()

    try:

        cursor.execute(sql)

        columns = [
            column[0]
            for column in cursor.description
        ]

        rows = cursor.fetchall()

        results = []

        for row in rows:

            record = {}

            for index, column in enumerate(
                columns
            ):

                record[column] = row[index]

            results.append(
                record
            )

        return results

    finally:

        cursor.close()
        connection.close()


def analyze_results(
    question,
    sql,
    results
):

    llm = create_llm()

    prompt = ChatPromptTemplate.from_template(
        """
You are a business data analysis agent.

Answer the user's question using ONLY
the SQL result below.

User question:
{question}

SQL result:
{results}

Rules:

1. Use only the SQL result.
2. Do not invent information.
3. Do not use outside knowledge.
4. Give a clear business-friendly answer.
5. If there are no results, say that no
   matching data was found.
6. Do not mention internal agents.
7. Do not mention SQL generation.
8. Do not mention prompts.
9. Do not return JSON.
10. Return ONLY the final answer.

Answer:
"""
    )

    messages = prompt.format_messages(
        question=question,
        results=str(results)
    )

    response = llm.invoke(
        messages
    )

    return extract_text(
        response
    )


def sql_agent(question):

    print(
        "[SQL Agent] Reading database schema..."
    )

    schema = get_database_schema()

    if not schema.strip():

        return (
            "I could not find any database "
            "schema information."
        )

    previous_sql = None
    error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        print(
            f"[SQL Agent] Attempt "
            f"{attempt}/{MAX_RETRIES}"
        )

        sql = generate_sql(
            question=question,
            schema=schema,
            previous_sql=previous_sql,
            error=error
        )

        print(
            "[SQL Agent] Generated SQL:"
        )

        print(sql)

        valid, validation_error = validate_sql(
            sql
        )

        if not valid:

            print(
                "[SQL Validator] SQL rejected:"
            )

            print(
                validation_error
            )

            previous_sql = sql
            error = validation_error

            continue

        print(
            "[SQL Validator] SQL is safe."
        )

        try:

            results = execute_sql(
                sql
            )

            print(
                "[SQL Agent] SQL executed successfully."
            )

            return analyze_results(
                question=question,
                sql=sql,
                results=results
            )

        except Exception as exc:

            print(
                "[SQL Agent] SQL execution failed:"
            )

            print(exc)

            previous_sql = sql
            error = str(exc)

    return (
        "I could not generate a valid SQL "
        "answer after multiple attempts."
    )