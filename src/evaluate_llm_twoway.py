
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "finguard_evaluation_reviewed_30.csv"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "finguard_evaluation_two_stage_predictions.csv"
)

MODEL = "openai/gpt-oss-20b"
LABELS = {"Informational", "Advisory", "Uncertain"}

load_dotenv(PROJECT_ROOT / ".env")

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY is missing from the project .env file.")

client = Groq()


def ask_model(prompt, text):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def classify_two_stage(text):
    # Stage 1: determine whether intent is clear from the text alone.
    stage1_prompt = """
You are assessing communicative intent in financial text.

Informational means the text describes or explains something without
recommending a financial action.

Advisory means the text recommends, encourages, discourages, or implies
a financial action.

Return JSON with exactly these fields:
{
  "intent_status": "clear" or "unclear",
  "reason": "brief explanation"
}

Choose "unclear" only when the text alone does not provide enough
evidence to distinguish informational from advisory intent. This
includes genuine ambiguity and insufficient context.

Do not treat a financial outcome being uncertain as uncertain intent.
Do not assume every short sentence is unclear.
Do not use any reference label.
Return JSON only.
"""

    first_raw = ask_model(stage1_prompt, text)

    try:
        first = json.loads(first_raw)
    except json.JSONDecodeError:
        return {
            "llm_prediction": "INVALID",
            "uncertainty_reason": "",
            "stage1_output": first_raw,
            "stage2_output": "",
            "prediction_error": "Stage 1 returned invalid JSON",
        }

    status = first.get("intent_status")
    reason = str(first.get("reason", ""))

    if status == "unclear":
        return {
            "llm_prediction": "Uncertain",
            "uncertainty_reason": reason,
            "stage1_output": first_raw,
            "stage2_output": "",
            "prediction_error": "",
        }

    if status != "clear":
        return {
            "llm_prediction": "INVALID",
            "uncertainty_reason": reason,
            "stage1_output": first_raw,
            "stage2_output": "",
            "prediction_error": "Stage 1 returned an unexpected status",
        }

    # Stage 2: classify only texts whose intent was judged clear.
    stage2_prompt = """
Classify the communicative intent of the supplied financial text.

Informational: describes or explains a fact, feature, concept, or risk
without recommending a financial action.

Advisory: recommends, encourages, discourages, or implies a financial
action or choice.

The text has already been assessed as having clear intent. Choose the
label best supported by its wording.

Return JSON with exactly these fields:
{
  "label": "Informational" or "Advisory",
  "reason": "brief explanation"
}

Do not use any reference label. Return JSON only.
"""

    second_raw = ask_model(stage2_prompt, text)

    try:
        second = json.loads(second_raw)
    except json.JSONDecodeError:
        return {
            "llm_prediction": "INVALID",
            "uncertainty_reason": reason,
            "stage1_output": first_raw,
            "stage2_output": second_raw,
            "prediction_error": "Stage 2 returned invalid JSON",
        }

    label = second.get("label")

    if label not in {"Informational", "Advisory"}:
        label = "INVALID"
        error = "Stage 2 returned an unexpected label"
    else:
        error = ""

    return {
        "llm_prediction": label,
        "uncertainty_reason": reason,
        "stage1_output": first_raw,
        "stage2_output": second_raw,
        "prediction_error": error,
    }


if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

required = {"example_id", "text", "label"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

if len(df) != 30:
    raise ValueError(f"Expected 30 evaluation examples, found {len(df)}")

if not df["label"].isin(LABELS).all():
    raise ValueError("Unexpected reference labels found.")

if df["text"].isna().any():
    raise ValueError("Missing text values found.")

records = []

for i, row in df.iterrows():
    print(f"Evaluating {i + 1}/{len(df)}: {row['example_id']}")

    try:
        result = classify_two_stage(str(row["text"]))
    except Exception as exc:
        result = {
            "llm_prediction": "ERROR",
            "uncertainty_reason": "",
            "stage1_output": "",
            "stage2_output": "",
            "prediction_error": str(exc),
        }

    records.append(result)

results = pd.concat(
    [df.reset_index(drop=True), pd.DataFrame(records)],
    axis=1,
)

results["correct"] = results["label"] == results["llm_prediction"]

# Save to a new file; preserve baseline and source dataset.
results.to_csv(OUTPUT_PATH, index=False)

valid = results[results["llm_prediction"].isin(LABELS)]

print("\n--- Two-Stage Evaluation ---")
print(f"Total examples: {len(results)}")
print(f"Valid predictions: {len(valid)}")
print(f"Invalid predictions: {(results['llm_prediction'] == 'INVALID').sum()}")
print(f"API errors: {(results['llm_prediction'] == 'ERROR').sum()}")

if len(valid):
    print(f"Accuracy on valid predictions: {valid['correct'].mean():.2%}")

    print("\nPer-class results:")
    print(
        valid.groupby("label").agg(
            total=("label", "size"),
            correct=("correct", "sum"),
        ).to_string()
    )

    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(
        pd.crosstab(
            valid["label"],
            valid["llm_prediction"],
            rownames=["Actual"],
            colnames=["Predicted"],
        ).reindex(
            index=["Informational", "Advisory", "Uncertain"],
            columns=["Informational", "Advisory", "Uncertain"],
            fill_value=0,
        )
    )

print(f"\nSaved predictions to: {OUTPUT_PATH}")
