from agents.retrieval_agent import retrieval_agent


question = input("Question: ")


result = retrieval_agent(
    question
)


print()

print("Number of documents:")

print(
    len(
        result.get(
            "documents",
            []
        )
    )
)


print()

print("Retrieved Context:")

print("=" * 60)

print(
    result.get(
        "context",
        ""
    )
)

print("=" * 60)