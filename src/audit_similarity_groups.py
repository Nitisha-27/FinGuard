
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "finguard_grouped.csv"

df = pd.read_csv(DATA_PATH)

print("=== DATASET SUMMARY ===")
print("File:", DATA_PATH)
print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print("Unique examples:", df["example_id"].nunique())
print("Similarity groups:", df["similarity_group"].nunique())

sizes = df.groupby("similarity_group").size()

print("\n=== GROUP SIZES ===")
print("Groups with 1 example:", int((sizes == 1).sum()))
print("Groups with 2+ examples:", int((sizes >= 2).sum()))
print("Largest group:", int(sizes.max()))
print("\nLargest 10 groups:")
print(sizes.sort_values(ascending=False).head(10).to_string())

print("\n=== LABEL DISTRIBUTION ===")
print(df["label"].value_counts().to_string())

label_counts = pd.crosstab(
    df["similarity_group"],
    df["label"]
)

mixed = label_counts[(label_counts > 0).sum(axis=1) > 1]

print("\n=== MIXED-LABEL GROUPS ===")
print("Number of groups containing multiple labels:", len(mixed))

for group_id in mixed.index:
    group = df[df["similarity_group"] == group_id]
    print(f"\nGroup {group_id}:")
    print(group[["example_id", "label", "text"]].to_string(index=False))

print("\n=== REVIEW STATUS ===")
if "review_status" in df.columns:
    print(df["review_status"].value_counts(dropna=False).to_string())

print("\nAudit finished. No rows were modified.")
