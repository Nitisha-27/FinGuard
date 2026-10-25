
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "grouped_oof_predictions.csv"

df = pd.read_csv(DATA_PATH)

required = {
    "example_id",
    "text",
    "label",
    "predicted_label",
    "similarity_group",
    "is_correct",
}

missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

# Normalize text only for identifying exact repetitions.
df["normalized_text"] = (
    df["text"]
    .fillna("")
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

print("=== DATASET AND ERROR SUMMARY ===")
print("Rows:", len(df))
print("Incorrect predictions:", int((~df["is_correct"].astype(bool)).sum()))
print("Unique normalized texts:", df["normalized_text"].nunique())

print("\n=== EXACT TEXT REPETITIONS ===")
text_counts = df["normalized_text"].value_counts()
repeated_texts = text_counts[text_counts > 1]

print("Texts appearing more than once:", len(repeated_texts))
print("Rows belonging to repeated texts:", int(repeated_texts.sum()))

print("\n=== ERROR RATE BY SIMILARITY GROUP ===")
group_summary = (
    df.groupby("similarity_group")
    .agg(
        examples=("example_id", "size"),
        errors=("is_correct", lambda s: int((~s.astype(bool)).sum())),
        labels=("label", "nunique"),
    )
)

group_summary["error_rate"] = (
    group_summary["errors"] / group_summary["examples"]
)

print(group_summary.sort_values(
    ["error_rate", "examples"],
    ascending=[False, False]
).head(15).to_string())

print("\n=== MOST FREQUENT ERROR TRANSITIONS ===")
errors = df[~df["is_correct"].astype(bool)].copy()

transitions = (
    errors.groupby(["label", "predicted_label"])
    .size()
    .sort_values(ascending=False)
)

print(transitions.to_string())

print("\n=== REPEATED TEXTS WITH INCONSISTENT LABELS ===")
label_counts = (
    df.groupby("normalized_text")["label"]
    .nunique()
)

inconsistent_texts = label_counts[label_counts > 1].index
inconsistent = df[df["normalized_text"].isin(inconsistent_texts)]

if inconsistent.empty:
    print("No exact repeated texts have different labels.")
else:
    print(
        inconsistent[
            ["example_id", "label", "predicted_label", "text"]
        ].sort_values("text").to_string(index=False)
    )

print("\nAudit complete. No dataset rows were modified.")
