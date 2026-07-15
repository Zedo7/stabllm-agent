"""
Measures how stable an LLM's answers are to the same factual question when
phrased differently.

Uses the real Anthropic API. Setup:
  - pip install anthropic python-dotenv
  - put your key in a .env file:  ANTHROPIC_API_KEY=sk-ant-...
  - run:  python stability_check.py

Swap models by changing MODEL below.
"""

import os
from collections import Counter

import anthropic
from dotenv import load_dotenv

load_dotenv()  # loads ANTHROPIC_API_KEY from .env into the environment

MODEL = "claude-haiku-4-5"

QUESTION = "What is the capital of France?"

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def generate_paraphrases(question: str) -> list[str]:
    """Return 5 paraphrases of `question` with varied wording/tone."""
    prompt = (
        f"Generate 5 paraphrases of the following question. Vary the wording "
        f"and tone, but keep the exact same meaning. Return only the 5 "
        f"paraphrases, one per line, with no numbering or extra text.\n\n"
        f"Question: {question}"
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    text = next(b.text for b in response.content if b.type == "text")
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[:5]


def get_answer(question: str) -> str:
    """Return the model's answer to `question`."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=64,
        messages=[{"role": "user", "content": question}],
    )
    return next(b.text for b in response.content if b.type == "text").strip()


def main():
    questions = [QUESTION] + generate_paraphrases(QUESTION)
    answers = [get_answer(q) for q in questions]

    counts = Counter(answers)
    majority_answer, majority_count = counts.most_common(1)[0]
    consistency = majority_count / len(answers)

    print(f"Question: {QUESTION}\n")
    print("Answers across original + 5 paraphrases:")
    for q, a in zip(questions, answers):
        flag = "" if a == majority_answer else "  <-- DEVIATES"
        print(f"  [{a}]\n    {q}{flag}")

    print(f"\nMajority answer: {majority_answer!r} ({majority_count}/{len(answers)})")
    print(f"Consistency score: {consistency:.0%}")


if __name__ == "__main__":
    main()
