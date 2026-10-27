import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "finguard_evaluation_reviewed_30_revised.csv"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "finguard_final_predictions.csv"
)

MODEL = "openai/gpt-oss-120b"
LABELS = {"Informational", "Advisory", "Uncertain"}
LABEL_ORDER = ["Informational", "Advisory", "Uncertain"]
N_SAMPLES = 5

load_dotenv(PROJECT_ROOT / ".env")


def classification_prompt():
    """Deterministic classifier with few-shot Uncertain anchors."""
    return (
        "Classify the supplied financial text into exactly one of three labels.\n\n"

        "LABEL DEFINITIONS\n"
        "Informational: States or explains a financial fact, definition, concept, "
        "risk, comparison, product feature, or neutral assessment without directing, "
        "recommending, encouraging, discouraging, or implicitly favouring a financial "
        "action or choice.\n\n"

        "Advisory: Explicitly or implicitly recommends, encourages, discourages, or "
        "favours a financial action or choice — even softly ('consider', 'you may want "
        "to', 'it is worth reviewing'). A clear recommendation remains Advisory even "
        "if phrased tentatively.\n\n"

        "Uncertain: Use when a reasonable expert could read the text as EITHER a "
        "neutral observation OR an implicit recommendation, and the supplied context "
        "cannot resolve which reading is intended. Uncertain is a deliberate label — "
        "do not avoid it.\n\n"

        "CALIBRATION EXAMPLES FOR UNCERTAIN\n"
        "These are Uncertain because neither reading is clearly dominant:\n"
        "- 'The current allocation is worth a closer look.' "
        "[neutral observation OR implicit review recommendation]\n"
        "- 'The lower-cost fund may be the more suitable option here.' "
        "[analytical finding OR implicit fund preference]\n"
        "- 'There are advantages to retaining the existing position.' "
        "[neutral fact enumeration OR implicit hold recommendation]\n"
        "- 'The recent decline makes the timing more significant.' "
        "[factual observation about timing OR implied prompt to act now]\n"
        "- 'The fee difference is meaningful in this comparison.' "
        "[reporting an analytical finding OR steering toward lower-fee product]\n\n"

        "DECISION GUIDE\n"
        "1. Does the text clearly direct or steer a financial action, even softly? "
        "→ Advisory\n"
        "2. Does it report facts or neutral analysis with no steering? "
        "→ Informational\n"
        "3. Could a reasonable expert plausibly read it either way, unresolved by "
        "context? → Uncertain\n\n"

        "Do not avoid Uncertain. Do not use it merely because the text is short.\n\n"

        "Return a JSON object with exactly these two keys:\n"
        '{"label": "Informational|Advisory|Uncertain", '
        '"explanation": "One sentence naming the specific word or phrase that '
        'determined the classification, and why."}'
    )


def confidence_prompt():
    """Simpler prompt used only for vote sampling — no JSON, no explanation."""
    return (
        "Classify the financial text as exactly one of: "
        "Informational, Advisory, or Uncertain.\n\n"

        "Informational: States financial facts or neutral analysis without steering "
        "any action.\n"
        "Advisory: Recommends, encourages, or steers a financial action, even softly.\n"
        "Uncertain: Both interpretations are plausible and context cannot resolve them.\n\n"

        "Uncertain examples: 'The current allocation is worth a closer look.' / "
        "'The lower-cost fund may be the more suitable option here.' / "
        "'There are advantages to retaining the existing position.'\n\n"

        "Return exactly one word: Informational, Advisory, or Uncertain."
    )


def classify_deterministic(client, text, context):
    """Single call at temperature=0. Returns (label, explanation)."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": classification_prompt()},
            {"role": "user", "content": f"Context: {context}\n\nText: {text}"},
        ],
        temperature=0,
    )
    raw = (response.choices[0].message.content or "").strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(raw)
        label = str(parsed.get("label", "")).strip()
        explanation = str(parsed.get("explanation", "")).strip()
        for l in LABEL_ORDER:
            if l.casefold() == label.casefold():
                return l, explanation
    except (json.JSONDecodeError, AttributeError):
        # Fallback: try plain text
        for l in LABEL_ORDER:
            if l.casefold() in raw.casefold():
                return l, ""

    return "INVALID", raw


def classify_vote(client, text, context):
    """Single stochastic call at temperature=0.7 for vote sampling."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": confidence_prompt()},
            {"role": "user", "content": f"Context: {context}\n\nText: {text}"},
        ],
        temperature=0.7,
    )
    raw = (response.choices[0].message.content or "").strip()
    for l in LABEL_ORDER:
        if l.casefold() == raw.casefold():
            return l
    return "INVALID"


def get_vote_confidence(client, text, context, n=N_SAMPLES):
    """
    Run N stochastic samples and return:
      - vote_majority: most-voted label (or TIE)
      - vote_confidence: fraction of votes for the majority
      - vote_distribution: full vote counts
      - vote_entropy: normalised entropy (0=certain, 1=maximally uncertain)
    """
    votes = [classify_vote(client, text, context) for _ in range(n)]
    valid_votes = [v for v in votes if v in LABELS]

    if not valid_votes:
        return "INVALID", None, {}, None

    counts = Counter(valid_votes)
    top_count = max(counts.values())
    top_labels = sorted(l for l, c in counts.items() if c == top_count)

    majority = "TIE" if len(top_labels) > 1 else top_labels[0]
    confidence = top_count / len(valid_votes)

    # Entropy over vote distribution (normalised by log2(3) for 3 classes)
    import math
    entropy = 0.0
    for c in counts.values():
        p = c / len(valid_votes)
        entropy -= p * math.log2(p)
    normalised_entropy = entropy / math.log2(3) if len(LABELS) > 1 else 0.0

    dist = {l: counts.get(l, 0) for l in LABEL_ORDER}

    return majority, confidence, dist, round(normalised_entropy, 4)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY missing.")

    df = pd.read_csv(DATA_PATH)

    if "context" not in df.columns:
        df["context"] = "No additional context provided."
    else:
        df["context"] = df["context"].fillna("No additional context provided.")

    client = Groq()
    rows = []

    for i, row in df.iterrows():
        print(f"[{i+1}/{len(df)}] {row['example_id']}", end=" ... ", flush=True)

        text, context = str(row["text"]), str(row["context"])

        # Step 1: deterministic prediction + explanation
        try:
            label, explanation = classify_deterministic(client, text, context)
        except Exception as e:
            label, explanation = "ERROR", str(e)

        # Step 2: stochastic vote confidence
        try:
            _, vote_conf, vote_dist, entropy = get_vote_confidence(
                client, text, context
            )
        except Exception as e:
            vote_conf, vote_dist, entropy = None, {}, None

        print(f"{label} (conf={vote_conf})")

        rows.append({
            "example_id": row["example_id"],
            "text": text,
            "context": context,
            "label": row["label"],
            "llm_prediction": label,
            "explanation": explanation,
            "vote_confidence": vote_conf,
            "vote_entropy": entropy,
            "votes_informational": vote_dist.get("Informational", 0),
            "votes_advisory": vote_dist.get("Advisory", 0),
            "votes_uncertain": vote_dist.get("Uncertain", 0),
        })

    results = pd.DataFrame(rows)
    results["correct"] = results["label"] == results["llm_prediction"]
    results.to_csv(OUTPUT_PATH, index=False)

    valid = results[results["llm_prediction"].isin(LABELS)]

    print("\n--- Final Evaluation Summary ---")
    print(f"Total rows:       {len(results)}")
    print(f"Valid:            {len(valid)}")
    print(f"Accuracy:         {valid['correct'].mean():.2%}")

    print("\nPer-class results:")
    print(
        valid.groupby("label").agg(
            total=("label", "size"),
            correct=("correct", "sum"),
            mean_vote_confidence=("vote_confidence", "mean"),
            mean_vote_entropy=("vote_entropy", "mean"),
        ).reindex(LABEL_ORDER).to_string()
    )

    print("\nConfusion matrix:")
    print(
        pd.crosstab(
            valid["label"], valid["llm_prediction"],
            rownames=["Actual"], colnames=["Predicted"],
        ).reindex(index=LABEL_ORDER, columns=LABEL_ORDER, fill_value=0)
    )

    print("\nKey finding — vote entropy by human label:")
    print(
        "Higher entropy = model is more uncertain across samples.\n"
        "If Uncertain examples show higher entropy than clear examples,\n"
        "that IS the correlation your research report needs.\n"
    )
    print(valid.groupby("label")["vote_entropy"].describe().reindex(LABEL_ORDER))

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()