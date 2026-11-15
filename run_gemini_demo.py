
from src.llm import generate_answer


question = "Explain Artificial Intelligence in simple words."

answer = generate_answer(question)

print("\nGemini Response:\n")
print(answer)