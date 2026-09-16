from agents.web_agent import web_agent


question = input(
    "Enter a web question: "
)

print()

print("=" * 60)

answer = web_agent(
    question
)

print(answer)

print("=" * 60)