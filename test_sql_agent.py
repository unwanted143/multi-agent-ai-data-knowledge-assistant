from agents.sql_agent import sql_agent


def main():

    print()
    print("==========================================")
    print("             SQL AGENT TEST")
    print("==========================================")
    print()

    question = input("Business Question: ").strip()

    if not question:
        return

    try:

        answer = sql_agent(question)

        print()
        print("Answer:")
        print(answer)
        print()

    except Exception as e:

        print()
        print("ERROR:")
        print(str(e))
        print()


if __name__ == "__main__":
    main()