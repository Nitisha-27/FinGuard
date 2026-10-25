
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "finguard_dataset.csv"

df = pd.read_csv(DATA_PATH)

print("========== FINGUARD DATASET AUDIT ==========")

# 1. Basic overview
print("\nTotal rows:", len(df))
print("Total columns:", len(df.columns))

# 2. Missing values
print("\nMissing values by column:")
print(df.isna().sum().to_string())

# 3. Duplicate IDs and texts
print("\nDuplicate IDs:", df["example_id"].duplicated().sum())
print("Duplicate texts:", df["text"].str.strip().str.lower().duplicated().sum())

# 4. Label distribution
print("\nLabel distribution:")
print(df["label"].value_counts(dropna=False).to_string())

# 5. Source distribution
print("\nSource distribution:")
print(df["source_type"].value_counts(dropna=False).to_string())

# 6. Review-status distribution
print("\nReview status:")
print(df["review_status"].value_counts(dropna=False).to_string())

# 7. Text length
df["text_length"] = df["text"].astype(str).str.split().str.len()
print("\nText length in words:")
print(df["text_length"].describe().round(2).to_string())

# 8. Most frequent exact words (simple diagnostic)
all_words = (
    df["text"]
    .dropna()
    .str.lower()
    .str.findall(r"\b[a-z]+\b")
    .explode()
)

print("\nTen most frequent words:")
print(all_words.value_counts().head(10).to_string())

# 9. Basic integrity checks
expected_labels = {"Informational", "Advisory", "Uncertain"}
actual_labels = set(df["label"].dropna().unique())

print("\n========== CHECKS ==========")
print("Exactly 300 rows:", len(df) == 300)
print("Unique IDs:", df["example_id"].is_unique)
print("Expected labels present:", actual_labels == expected_labels)
print("All examples still candidates:", df["review_status"].eq("candidate").all())

print("\nAudit complete. These checks assess data quality,")
print("not whether every financial label is correct.")
