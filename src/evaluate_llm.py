
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "finguard_published.csv"

load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in project-root .env file.")

client = Groq(api_key=api_key)
MODEL = "openai/gpt-oss-20b"

LABELS = {"Informational", "Advisory", "Uncertain"}


def classify(text):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify financial text into exactly one label.\n\n"
                    "Informational: The text primarily describes, explains, "
                    "or reports a financial concept, product feature, risk, "
                    "or historical fact. It does not recommend an action.\n\n"
                    "Advisory: The text recommends, encourages, discourages, "
                    "or suggests a financial action or choice. Indirect "
                    "recommendations count, such as 'worth considering' or "
                    "'investors should review'.\n\n"
                    "Uncertain: The text has materially ambiguous communicative "
                    "intent. It could reasonably be read as either informational "
                    "or advisory, and the wording alone does not resolve that "
                    "distinction. Use this label when that ambiguity is central, "
                    "not merely because the text uses words such as 'may' or 'could'.\n\n"
                    "Classify only the supplied financial text. Do not use the "
                    "existing dataset label as the answer. Return exactly one "
                    "label: Informational, Advisory, or Uncertain."
                ),
            },
            {"role": "user", "content": text},
        ],
        temperature=0,
    )

    result = response.choices[0].message.content.strip()

    for label in LABELS:
        if result.casefold() == label.casefold():
            return label

    print(f"Unexpected model output: {result!r}")
    return "INVALID"



df = pd.read_csv(DATA_PATH)

required = {"text", "label"}
if not required.issubset(df.columns):
    raise ValueError(f"Dataset must contain columns: {required}")


sample = (
    df.groupby("label", group_keys=False)
      .sample(n=10, random_state=123)
      .reset_index(drop=True)
)

predictions = []

for i, row in sample.iterrows():
    print(f"Classifying {len(predictions) + 1}/{len(sample)}...")
    try:
        prediction = classify(str(row["text"]))
    except Exception as exc:
        print(f"Request failed for row {i}: {type(exc).__name__}: {exc}")
        prediction = "ERROR"

    predictions.append(prediction)
    time.sleep(1)

sample["llm_prediction"] = predictions
sample["correct"] = (
    sample["label"].astype(str).str.strip().str.casefold()
    == sample["llm_prediction"].astype(str).str.casefold()
)

output_path = PROJECT_ROOT / "data" / "processed" / "llm_evaluation_30.csv"
sample.to_csv(output_path, index=False)

valid = sample[sample["llm_prediction"].isin(LABELS)]
print("\nResults")
print("Rows tested:", len(sample))
print("Valid predictions:", len(valid))
if len(valid):
    print("Accuracy on valid predictions:", round(valid["correct"].mean(), 3))

print("\nPredictions:")
print(sample[["example_id", "label", "llm_prediction", "correct"]].to_string(index=False))
print(f"\nSaved results to: {output_path}")
