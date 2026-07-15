"""
Measures how stable an LLM's answers are to the same factual question when
phrased differently.

Instead of random rephrasings, paraphrases follow a taxonomy of four
categories, each with five levels (from capstone research). This lets us see
*which kind* of phrasing variation destabilizes the answer.

Setup:
  - pip install anthropic python-dotenv
  - put your key in a .env file:  ANTHROPIC_API_KEY=sk-ant-...
  - run:  python stability_check.py

Swap models by changing MODEL below.
"""

import os
import re
import string
import sys
from collections import Counter

import anthropic
from dotenv import load_dotenv

# The taxonomy includes non-Latin languages (Chinese, Japanese); force UTF-8
# output so printing them doesn't crash on a legacy console (e.g. Windows cp1252).
sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()  # loads ANTHROPIC_API_KEY from .env into the environment

MODEL = "claude-haiku-4-5"

QUESTION = "What is the capital of France?"

# --- Paraphrase taxonomy -----------------------------------------------------
# Each category maps to five ordered levels. Edit freely: add/remove categories
# or levels here and the rest of the pipeline adapts automatically.
TAXONOMY = {
    "VOCABULARY": ["very simple", "simple", "medium", "advanced", "very advanced"],
    "TONE": ["very casual", "moderately casual", "neutral", "moderately formal", "very formal"],
    "GRAMMAR": ["simple", "compound", "complex", "passive voice", "inverted"],
    "LANGUAGE": ["Chinese", "Spanish", "French", "German", "Japanese"],
}

# How each category's level is phrased inside the paraphrase prompt. `{level}`
# is substituted with the specific level being applied.
CATEGORY_INSTRUCTIONS = {
    "VOCABULARY": "using {level} vocabulary (adjust word difficulty only)",
    "TONE": "in a {level} tone",
    "GRAMMAR": "using {level} sentence structure",
    "LANGUAGE": "translated into {level}",
}

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def generate_paraphrases(question: str, category: str, level: str) -> str:
    """Generate one paraphrase applying a single category+level variation.

    The prompt instructs the model to apply only that one variation while
    preserving the exact semantic meaning of the question.
    """
    instruction = CATEGORY_INSTRUCTIONS[category].format(level=level)
    prompt = (
        f"Rewrite the question below {instruction}.\n"
        f"Apply ONLY this one variation. Preserve the exact same semantic "
        f"meaning — it must still ask for exactly the same information. Do not "
        f"add, remove, or answer anything.\n"
        f"Return only the rewritten question, with no quotes, labels, or "
        f"explanation.\n\n"
        f"Question: {question}"
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return next(b.text for b in response.content if b.type == "text").strip()


def get_answer(question: str) -> str:
    """Return the model's answer, constrained to a direct English answer.

    The English requirement keeps LANGUAGE-variant answers comparable to the
    rest even though the question itself is in another language.
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=64,
        system=(
            "Answer with only the direct answer — a single word or short "
            "phrase. Always answer in English. No full sentences, no "
            "punctuation, no explanation."
        ),
        messages=[{"role": "user", "content": question}],
    )
    return next(b.text for b in response.content if b.type == "text").strip()


def normalize(answer: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace for comparison."""
    text = answer.lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def main():
    # Build every variant: the original plus all category x level combinations.
    variants = [("ORIGINAL", "-", QUESTION)]
    for category, levels in TAXONOMY.items():
        for level in levels:
            print(f"Generating {category} / {level} ...", flush=True)
            paraphrase = generate_paraphrases(QUESTION, category, level)
            variants.append((category, level, paraphrase))

    # Answer each variant.
    rows = []  # (category, level, paraphrase, answer, normalized_answer)
    for category, level, paraphrase in variants:
        print(f"Answering {category} / {level} ...", flush=True)
        answer = get_answer(paraphrase)
        rows.append((category, level, paraphrase, answer, normalize(answer)))

    # Overall majority is computed across every variant, including the original.
    normalized = [r[4] for r in rows]
    majority_answer, majority_count = Counter(normalized).most_common(1)[0]
    overall_consistency = majority_count / len(rows)

    # --- Summary table -------------------------------------------------------
    print("\n" + "=" * 100)
    print(f"Question: {QUESTION}")
    print(f"Model:    {MODEL}")
    print("=" * 100)
    header = f"{'CATEGORY':<10} {'LEVEL':<18} {'PARAPHRASE':<40} {'ANSWER':<14} DEVIATES"
    print(header)
    print("-" * 100)
    for category, level, paraphrase, answer, norm in rows:
        deviates = "" if norm == majority_answer else "YES"
        print(
            f"{category:<10} {level:<18} {truncate(paraphrase, 40):<40} "
            f"{truncate(answer, 14):<14} {deviates}"
        )

    # --- Consistency per category -------------------------------------------
    print("\n" + "-" * 100)
    print("Consistency by category (share of that category's variants matching the majority answer):")
    per_category = {}
    for category in TAXONOMY:
        cat_norms = [r[4] for r in rows if r[0] == category]
        matches = sum(1 for n in cat_norms if n == majority_answer)
        per_category[category] = matches / len(cat_norms)

    for category, score in sorted(per_category.items(), key=lambda kv: kv[1]):
        print(f"  {category:<10} {score:>5.0%}")

    worst = min(per_category, key=per_category.get)
    print(f"\nMost instability: {worst} ({per_category[worst]:.0%} consistent)")
    print(f"Majority answer:  {majority_answer!r} ({majority_count}/{len(rows)} variants)")
    print(f"Overall consistency: {overall_consistency:.0%}")


if __name__ == "__main__":
    main()
