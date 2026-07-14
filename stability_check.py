"""
Measures how stable an LLM's answers are to the same factual question when
phrased differently.

Currently uses MOCK generate_paraphrases()/get_answer() so the pipeline runs
end-to-end without an API key. To switch to real Anthropic API calls:
  - pip install anthropic
  - export ANTHROPIC_API_KEY=...
  - replace the bodies of generate_paraphrases()/get_answer() with the
    commented-out implementations below (same signatures, so nothing else
    in this file needs to change).
"""

import os
from collections import Counter

QUESTION = "What is the capital of France?"

# import anthropic
# client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
# MODEL = "claude-opus-4-8"


def generate_paraphrases(question: str) -> list[str]:
    """Return 5 paraphrases of `question` with varied wording/tone.

    MOCK: hardcoded reworded versions of the capital-of-France question.
    """
    return [
        "Which city serves as the capital of France?",
        "Can you tell me France's capital city?",
        "France's capital is which city?",
        "I'm curious, what's the capital city of France?",
        "Name the capital of France.",
    ]

    # Real implementation (swap in once ANTHROPIC_API_KEY is set):
    #
    # prompt = (
    #     f"Generate 5 paraphrases of the following question. Vary the "
    #     f"wording and tone, but keep the exact same meaning. Return only "
    #     f"the 5 paraphrases, one per line, no numbering.\n\nQuestion: {question}"
    # )
    # response = client.messages.create(
    #     model=MODEL,
    #     max_tokens=512,
    #     messages=[{"role": "user", "content": prompt}],
    # )
    # text = next(b.text for b in response.content if b.type == "text")
    # return [line.strip() for line in text.strip().splitlines() if line.strip()][:5]


def get_answer(question: str) -> str:
    """Return an answer to `question`.

    MOCK: mostly returns "Paris", but one phrasing deliberately returns a
    different answer so the consistency logic has something to flag.
    """
    mock_answers = {
        "What is the capital of France?": "Paris",
        "Which city serves as the capital of France?": "Paris",
        "Can you tell me France's capital city?": "Paris",
        "France's capital is which city?": "Lyon",  # deliberate deviation
        "I'm curious, what's the capital city of France?": "Paris",
        "Name the capital of France.": "Paris",
    }
    return mock_answers.get(question, "Unknown")

    # Real implementation (swap in once ANTHROPIC_API_KEY is set):
    #
    # response = client.messages.create(
    #     model=MODEL,
    #     max_tokens=64,
    #     messages=[{"role": "user", "content": question}],
    # )
    # return next(b.text for b in response.content if b.type == "text").strip()


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
        print(f"  [{a:10}] {q}{flag}")

    print(f"\nMajority answer: {majority_answer!r} ({majority_count}/{len(answers)})")
    print(f"Consistency score: {consistency:.0%}")


if __name__ == "__main__":
    main()
