"""
Measures how stable an LLM's answers are to the same factual question when
phrased differently.

Paraphrases follow a taxonomy of four categories, each with five levels (from
capstone research), so we can see *which kind* of phrasing variation
destabilizes the answer.

Two LLM-based verifier layers guard against measurement artifacts:
  1. PARAPHRASE VALIDATION (input side): each generated paraphrase is checked
     for semantic equivalence to the original before it enters the run.
  2. ANSWER EQUIVALENCE (output side): answers that string-matching flags as
     deviating are sent to an LLM judge; only genuinely different answers count
     as real instability.

The report shows raw string-match consistency AND verifier-adjusted
consistency, so you can see how much apparent instability is artifact vs real.

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
import unicodedata
from collections import Counter

import anthropic
from dotenv import load_dotenv

# The taxonomy includes non-Latin languages (Chinese, Japanese); force UTF-8
# output so printing them doesn't crash on a legacy console (e.g. Windows cp1252).
sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()  # loads ANTHROPIC_API_KEY from .env into the environment

MODEL = "claude-haiku-4-5"

QUESTION = "What is the capital of France?"

MAX_PARAPHRASE_RETRIES = 2  # regeneration attempts if validation fails

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


# --- LLM helpers -------------------------------------------------------------
def _ask(prompt: str, *, system: str | None = None, max_tokens: int = 256) -> str:
    kwargs = {"model": MODEL, "max_tokens": max_tokens,
              "messages": [{"role": "user", "content": prompt}]}
    if system is not None:
        kwargs["system"] = system
    response = client.messages.create(**kwargs)
    return next(b.text for b in response.content if b.type == "text").strip()


def _yes(prompt: str) -> bool:
    """Ask a yes/no question and return True iff the model says yes."""
    return _ask(prompt, system="Answer with only YES or NO.", max_tokens=5).lower().startswith("yes")


# --- Paraphrase generation + validation (input side) -------------------------
def generate_paraphrases(question: str, category: str, level: str) -> str:
    """Generate one paraphrase applying a single category+level variation."""
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
    return _ask(prompt)


def validate_paraphrase(original: str, paraphrase: str) -> bool:
    """LLM check: is `paraphrase` semantically equivalent to `original`?

    Equivalent means the same single question asking for exactly the same
    information — no added or removed sub-questions, no meaning drift. A
    different language or writing style is fine as long as the meaning holds.
    """
    prompt = (
        f"Original question:\n{original}\n\n"
        f"Candidate paraphrase:\n{paraphrase}\n\n"
        f"Is the candidate semantically equivalent to the original? It must be "
        f"a single question asking for exactly the same information, with no "
        f"added or removed sub-questions and no drift in meaning. A different "
        f"language, vocabulary, tone, or grammar is acceptable as long as the "
        f"meaning is identical. Answer YES or NO."
    )
    return _yes(prompt)


def make_validated_paraphrase(question, category, level):
    """Generate a paraphrase, retrying if validation fails.

    Returns (paraphrase, valid, attempts, rejected) where `rejected` is the
    list of paraphrases that failed validation along the way.
    """
    rejected = []
    paraphrase = ""
    for attempt in range(1, MAX_PARAPHRASE_RETRIES + 2):  # 1 initial + N retries
        paraphrase = generate_paraphrases(question, category, level)
        if validate_paraphrase(question, paraphrase):
            return paraphrase, True, attempt, rejected
        rejected.append(paraphrase)
    return paraphrase, False, MAX_PARAPHRASE_RETRIES + 1, rejected


# --- Answering + equivalence judging (output side) ---------------------------
def get_answer(question: str) -> str:
    """Return the model's answer, constrained to a direct English answer.

    The English requirement keeps LANGUAGE-variant answers comparable to the
    rest even though the question itself is in another language.
    """
    return _ask(
        question,
        system=(
            "Answer with only the direct answer — a single word or short "
            "phrase. Always answer in English. No full sentences, no "
            "punctuation, no explanation."
        ),
        max_tokens=64,
    )


def answers_equivalent(answer_a: str, answer_b: str) -> bool:
    """LLM judge: do two answers convey the same factual answer?"""
    prompt = (
        f"Two answers were given to the same factual question.\n"
        f"Answer A: {answer_a}\n"
        f"Answer B: {answer_b}\n\n"
        f"Do they convey the same factual answer? Ignore differences in "
        f"wording, language, spelling, capitalization, or extra detail — judge "
        f"only whether they mean the same thing. Answer YES or NO."
    )
    return _yes(prompt)


def normalize(answer: str) -> str:
    """Normalize an answer for comparison.

    Lowercases, strips diacritics (so "París" matches "paris" — LANGUAGE
    variants are judged on the answer, not local spelling), removes
    punctuation, and collapses whitespace.
    """
    # Decompose accented chars into base + combining marks, then drop the marks.
    decomposed = unicodedata.normalize("NFKD", answer)
    without_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    text = without_accents.lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def main():
    # --- Build variants with input-side validation ---------------------------
    variants = [{"category": "ORIGINAL", "level": "-", "paraphrase": QUESTION,
                 "valid": True, "attempts": 0}]
    validation_failures = []
    for category, levels in TAXONOMY.items():
        for level in levels:
            print(f"Generating + validating {category} / {level} ...", flush=True)
            paraphrase, valid, attempts, rejected = make_validated_paraphrase(
                QUESTION, category, level)
            variants.append({"category": category, "level": level,
                             "paraphrase": paraphrase, "valid": valid,
                             "attempts": attempts})
            if not valid:
                validation_failures.append((category, level, paraphrase, rejected))
                print(f"  ! validation FAILED after {attempts} attempts", flush=True)

    # --- Answer every variant ------------------------------------------------
    for v in variants:
        print(f"Answering {v['category']} / {v['level']} ...", flush=True)
        v["answer"] = get_answer(v["paraphrase"])
        v["norm"] = normalize(v["answer"])

    # --- Raw string-match consistency ---------------------------------------
    norms = [v["norm"] for v in variants]
    majority_answer, majority_count = Counter(norms).most_common(1)[0]
    n = len(variants)
    raw_consistency = majority_count / n

    # A representative raw answer from the majority class, for the judge.
    majority_raw = next(v["answer"] for v in variants if v["norm"] == majority_answer)

    # --- Output-side equivalence judging for flagged deviations -------------
    for v in variants:
        v["string_deviates"] = v["norm"] != majority_answer
        if v["string_deviates"]:
            print(f"Judging deviation: {v['category']} / {v['level']} "
                  f"({v['answer']!r} vs {majority_raw!r}) ...", flush=True)
            v["judged_equivalent"] = answers_equivalent(v["answer"], majority_raw)
        else:
            v["judged_equivalent"] = None
        # Real instability = string-deviates AND judge says they differ.
        v["real_deviation"] = v["string_deviates"] and not v["judged_equivalent"]

    real_deviations = sum(1 for v in variants if v["real_deviation"])
    adjusted_consistency = (n - real_deviations) / n

    # --- Summary table -------------------------------------------------------
    print("\n" + "=" * 104)
    print(f"Question: {QUESTION}")
    print(f"Model:    {MODEL}")
    print("=" * 104)
    print(f"{'CATEGORY':<10} {'LEVEL':<18} {'PARAPHRASE':<38} {'ANSWER':<12} "
          f"{'VALID':<6} DEVIATION")
    print("-" * 104)
    for v in variants:
        if v["category"] == "ORIGINAL":
            valid_col = "-"
        else:
            valid_col = "ok" if v["valid"] else "FAIL"
        if not v["string_deviates"]:
            deviation = ""
        elif v["real_deviation"]:
            deviation = "REAL"
        else:
            deviation = "artifact"  # string-deviated but judge ruled equivalent
        print(f"{v['category']:<10} {v['level']:<18} "
              f"{truncate(v['paraphrase'], 38):<38} {truncate(v['answer'], 12):<12} "
              f"{valid_col:<6} {deviation}")

    # --- Per-category consistency (raw vs adjusted) --------------------------
    print("\n" + "-" * 104)
    print("Consistency by category  (raw = string match; adjusted = after answer-equivalence judge):")
    print(f"  {'CATEGORY':<12} {'RAW':>6} {'ADJUSTED':>10}")
    per_category = {}
    for category in TAXONOMY:
        cat = [v for v in variants if v["category"] == category]
        raw_match = sum(1 for v in cat if not v["string_deviates"])
        adj_match = sum(1 for v in cat if not v["real_deviation"])
        per_category[category] = (raw_match / len(cat), adj_match / len(cat))
    for category, (raw_pct, adj_pct) in sorted(per_category.items(), key=lambda kv: kv[1][1]):
        print(f"  {category:<12} {raw_pct:>5.0%} {adj_pct:>10.0%}")

    # --- Paraphrase validation summary --------------------------------------
    generated = n - 1  # exclude the original
    validated_ok = generated - len(validation_failures)
    print("\n" + "-" * 104)
    print(f"Paraphrase validation: {validated_ok}/{generated} passed "
          f"(≤{MAX_PARAPHRASE_RETRIES} retries each).")
    for category, level, paraphrase, rejected in validation_failures:
        print(f"  FAILED  {category} / {level}: kept {paraphrase!r} "
              f"after {len(rejected)} rejected attempt(s)")

    # --- Headline numbers ----------------------------------------------------
    artifacts = sum(1 for v in variants
                    if v["string_deviates"] and not v["real_deviation"])
    print("\n" + "-" * 104)
    print(f"Majority answer:              {majority_answer!r} ({majority_count}/{n} by string match)")
    print(f"Raw consistency (string):     {raw_consistency:.0%}")
    print(f"Adjusted consistency (judge): {adjusted_consistency:.0%}")
    print(f"Apparent deviations:          {n - majority_count}  "
          f"({artifacts} artifact, {real_deviations} real)")


if __name__ == "__main__":
    main()
