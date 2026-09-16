from agents.graph import create_rag_graph


# =========================================================
# CREATE APPLICATION
# =========================================================

app = create_rag_graph()


# =========================================================
# MAIN APPLICATION
# =========================================================

def main():

    print()
    print("==============================================")
    print("       SUPERVISOR MULTI-AGENT AI")
    print("==============================================")
    print()

    print("Architecture:")
    print()
    print("                    User")
    print("                     |")
    print("                     v")
    print("               Supervisor")
    print("                     |")
    print("          +----------+----------+")
    print("          |          |          |")
    print("         RAG        SQL        WEB")
    print("          |          |          |")
    print("          v          v          |")
    print("     Retrieval    SQL Agent     |")
    print("       Agent         |          |")
    print("          |          v          |")
    print("          v       SQL Server    |")
    print("     Analysis         |         |")
    print("       Agent          |         |")
    print("          |           |         |")
    print("          +-----------+---------+")
    print("                      |")
    print("                      v")
    print("                    Answer")
    print()

    print("Supported:")
    print("  • Questions about local documents")
    print("  • Questions about SQL Server data")
    print()

    print("Type 'exit' to quit.")
    print()

    while True:

        # -------------------------------------------------
        # GET USER QUESTION
        # -------------------------------------------------

        question = input("You: ").strip()

        if not question:
            continue

        # -------------------------------------------------
        # EXIT
        # -------------------------------------------------

        if question.lower() == "exit":

            print()
            print("Goodbye!")

            break

        try:

            # -------------------------------------------------
            # INITIAL STATE
            # -------------------------------------------------

            initial_state = {

                 "question": question,

                 "route": "",

                "context": "",

                 "answer": "",

                "validation": "",

                 "retry_count": 0
            }

            # -------------------------------------------------
            # RUN LANGGRAPH
            # -------------------------------------------------

            result = app.invoke(
                initial_state
            )

            # -------------------------------------------------
            # GET FINAL ANSWER
            # -------------------------------------------------

            answer = result.get(
                "answer",
                "I could not find the answer in the provided documents."
            )

            # -------------------------------------------------
            # DISPLAY ONLY THE ANSWER
            # -------------------------------------------------

            print()
            print(answer)
            print()

        except Exception as e:

            print()
            print(
                "Error while processing your question:"
            )
            print(str(e))
            print()


# =========================================================
# START APPLICATION
# =========================================================

if __name__ == "__main__":

    main()