# StabLLM Agent

## What this is

A stability evaluation framework for LLMs: it measures whether a model gives
consistent answers when the same question is paraphrased. Built on the
methodology from my NYU MSDS capstone (*StabLLM: Evaluating and Promoting the
Stability of LLMs*).

## Method

- **Paraphrase taxonomy:** 4 categories × 5 levels — vocabulary complexity,
  tone, grammatical restructuring, and multilingual translation — giving 20
  variants plus the original per question.
- **Doer–Verifier architecture:** the model under test answers the questions; a
  separate, stronger verifier model handles paraphrase generation,
  semantic-equivalence validation, and answer judging.
- **Model decoupling:** `MODEL` (under test) and `VERIFIER_MODEL` are
  independent variables, so the judge never grades its own work. This makes
  cross-model comparison methodologically valid.
- **Two verifier layers:**
  1. *Input-side* — validates that each paraphrase is semantically equivalent to
     the original before it enters the pipeline; regenerates on failure (up to 2
     retries) and logs the failure.
  2. *Output-side* — when string matching flags a deviation, an LLM judge decides
     whether it is a real difference or a measurement artifact.
- **Dual reporting:** raw string-match consistency and verifier-adjusted
  consistency are reported side by side, with an artifact/real breakdown.

### Reproducibility

- `temperature=0` is set on every call whose model accepts it. Both `MODEL`
  (`claude-haiku-4-5`) and `VERIFIER_MODEL` (`claude-sonnet-4-5`) currently do,
  so it applies throughout. (Newer models such as Sonnet 5 reject an explicit
  temperature and would run at their default; the code omits it for those.)
- This makes all metrics and validation verdicts stable across runs, but it is
  **not** a bit-for-bit guarantee. Across two runs, 1 of 40 paraphrases differed
  by a single word (GRAMMAR / passive voice: *"called"* vs *"known as"*).
  `temperature=0` makes sampling greedy, but server-side batching, floating-point
  non-associativity, and MoE routing still allow token-level variation.
- The fix for true reproducibility (not implemented, scoped out): cache the
  generated paraphrase set to disk and load it on subsequent runs, freezing the
  input side regardless of API non-determinism.

## Findings

1. **Verifier decoupling matters.** With `MODEL` = `claude-haiku-4-5` under test
   and `VERIFIER_MODEL` = `claude-sonnet-4-5` (both at `temperature=0`), the
   verifier rejects the paraphrase *"What is the main city of France?"* at
   **both** the very-simple and medium vocabulary levels — main city can mean
   largest city, not capital — so 18/20 paraphrases pass validation. Haiku,
   grading its own paraphrases, accepts that same drift. A stronger, decoupled
   verifier catches semantic drift that a self-grading weaker model lets through.
2. **Measurement artifacts dominate apparent instability.** In a sample run, raw
   consistency was 95% and verifier-adjusted was 100% — the single "deviation"
   was Haiku answering the Japanese variant as パリ, which is the same answer, not
   instability.
3. **Prompt specification quality drives paraphrase validity.** The vague
   instruction *"using compound sentence structure"* caused the generator to
   append a second question, failing validation 3/3 attempts every run.
   Replacing it with per-level instructions plus a worked example and an explicit
   one-question constraint fixed it — all five GRAMMAR levels now pass.
4. **Some perturbation levels are inherently unparaphrasable.** *"Very simple
   vocabulary"* consistently forces *capital* → *main city*, which the verifier
   correctly rejects as a meaning change (main city can mean largest city). This
   is not a tool bug — it suggests the low consistency observed at
   simple-vocabulary levels in the original capstone may partly reflect
   paraphrase invalidity rather than model instability. Left as an honest flagged
   failure.

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

Create a `.env` file in the repo root with your Anthropic API key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Run:

```bash
python stability_check.py
```

Configure the run by editing the variables at the top of `stability_check.py`:
`MODEL` (model under test), `VERIFIER_MODEL`, `TEMPERATURE`, `QUESTION`, and the
`TAXONOMY` dict.

## Status

Working prototype, tested on single hardcoded questions. Next steps: run it
across the full benchmark datasets used in the capstone (ARC-Challenge,
TruthfulQA, GSM8K, MMLU-medical), and add paraphrase caching for full
reproducibility.
