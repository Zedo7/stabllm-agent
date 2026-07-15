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
import re
import string
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
    """Return the model's answer to `question`, constrained to a direct answer."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=64,
        system=(
            "Answer with only the direct answer — a single word or short "
            "phrase. No full sentences, no punctuation, no explanation."
        ),
        messages=[{"role": "user", "content": question}],
    )
    return next(b.text for b in response.content if b.type == "text").strip()


def normalize(answer: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace for comparison."""
    text = answer.lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def main():
    questions = [QUESTION] + generate_paraphrases(QUESTION)
    answers = [get_answer(q) for q in questions]
    normalized = [normalize(a) for a in answers]

    counts = Counter(normalized)
    majority_answer, majority_count = counts.most_common(1)[0]
    consistency = majority_count / len(answers)

    print(f"Question: {QUESTION}\n")
    print("Answers across original + 5 paraphrases:")
    for q, a, n in zip(questions, answers, normalized):
        flag = "" if n == majority_answer else "  <-- DEVIATES"
        print(f"  [{a}]\n    {q}{flag}")

    print(f"\nMajority answer: {majority_answer!r} ({majority_count}/{len(answers)})")
    print(f"Consistency score: {consistency:.0%}")


if __name__ == "__main__":
    main()
