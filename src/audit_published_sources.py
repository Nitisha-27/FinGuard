
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "finguard_published.csv"
)

df = pd.read_csv(DATASET_PATH)

print("=== FinGuard Source Audit ===")
print("Total examples:", len(df))

print("\n1. Review status:")
print(df["review_status"].value_counts(dropna=False))

print("\n2. Source type:")
print(df["source_type"].value_counts(dropna=False))

print("\n3. Unique source references:")
print(df["source_reference"].nunique())

print("\n4. Examples by label and source type:")
print(pd.crosstab(df["label"], df["source_type"]))

print("\n5. Rows needing source verification:")
needs_review = df[
    df["review_status"].astype(str).str.contains(
        "candidate|verify|pending", case=False, na=False
    )
]
print("Count:", len(needs_review))
print(needs_review[["example_id", "source_reference", "review_status"]].head(10).to_string(index=False))

print("\nAudit complete. This checks metadata, not source validity.")
