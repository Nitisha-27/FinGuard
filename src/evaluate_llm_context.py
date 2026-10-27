
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "finguard_evaluation_reviewed_30.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "finguard_evaluation_reviewed_30_predictions.csv"

MODEL = "openai/gpt-oss-20b"

LABELS = {"Informational", "Advisory", "Uncertain"}

load_dotenv(PROJECT_ROOT / ".env")


def classify(text):
    prompt = (
        "Classify the supplied financial text into exactly one label.\n\n"
        "Informational: Explains or reports a financial concept, fact, "
        "product feature, or risk without recommending an action.\n\n"
        "Advisory: Recommends, encourages, discourages, or implies a "
        "financial action or choice.\n\n"
        "Uncertain: The text alone does not provide enough evidence to "
        "decide whether its intent is informational or advisory.\n\n"
        "Use only the supplied text. Do not use the dataset's existing "
        "label or rationale. Return exactly one label: Informational, "
        "Advisory, or Uncertain."
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
        temperature=0,
    )

    prediction = response.choices[0].message.content.strip()

    if prediction in LABELS:
        return prediction

    return "INVALID"


# Validate the input before calling the API
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Evaluation dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

required_columns = {"example_id", "text", "label"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

if df.empty:
    raise ValueError("The evaluation dataset is empty.")

if not df["label"].isin(LABELS).all():
    raise ValueError("The dataset contains labels outside the three expected classes.")

if df["text"].isna().any():
    raise ValueError("The dataset contains missing text values.")

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY is missing. Check the .env file.")

client = Groq()

predictions = []

for index, row in df.iterrows():
    print(f"Evaluating {index + 1}/{len(df)}: {row['example_id']}")

    try:
        prediction = classify(row["text"])
        error = ""
    except Exception as exc:
        prediction = "ERROR"
        error = str(exc)
        print(f"  API error: {error}")

    predictions.append({
        "llm_prediction": prediction,
        "prediction_error": error,
    })

results = pd.concat(
    [df.reset_index(drop=True), pd.DataFrame(predictions)],
    axis=1,
)

results["correct"] = results["label"] == results["llm_prediction"]

# Save predictions separately; do not overwrite the input dataset
results.to_csv(OUTPUT_PATH, index=False)

valid_results = results[results["llm_prediction"].isin(LABELS)]

print("\n--- Evaluation Summary ---")
print(f"Dataset rows: {len(results)}")
print(f"Valid predictions: {len(valid_results)}")
print(f"Invalid predictions: {(results['llm_prediction'] == 'INVALID').sum()}")
print(f"API errors: {(results['llm_prediction'] == 'ERROR').sum()}")

if len(valid_results):
    print(f"Accuracy on valid predictions: {valid_results['correct'].mean():.2%}")

    print("\nPer-class results:")
    print(
        valid_results.groupby("label").agg(
            total=("label", "size"),
            correct=("correct", "sum"),
        ).to_string()
    )

    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(
        pd.crosstab(
            valid_results["label"],
            valid_results["llm_prediction"],
            rownames=["Actual"],
            colnames=["Predicted"],
        ).reindex(
            index=["Informational", "Advisory", "Uncertain"],
            columns=["Informational", "Advisory", "Uncertain"],
            fill_value=0,
        )
    )

print(f"\nPredictions saved to: {OUTPUT_PATH}")
