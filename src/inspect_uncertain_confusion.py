
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "grouped_oof_predictions.csv"

df = pd.read_csv(DATA_PATH)

cases = df[
    (df["label"] == "Uncertain")
    & (df["predicted_label"] == "Informational")
].copy()

columns = [
    "example_id",
    "similarity_group",
    "text",
    "context",
    "linguistic_rationale",
    "source_type",
    "source_reference",
]

columns = [c for c in columns if c in cases.columns]

print("Uncertain predicted as Informational:", len(cases))
print("=" * 80)

for _, row in cases.iterrows():
    print(f"\nID: {row['example_id']}")
    print(f"Similarity group: {row['similarity_group']}")
    print(f"Text: {row['text']}")
    
    if "context" in row.index:
        print(f"Context: {row['context']}")
    if "linguistic_rationale" in row.index:
        print(f"Original rationale: {row['linguistic_rationale']}")
    if "source_type" in row.index:
        print(f"Source type: {row['source_type']}")
    if "source_reference" in row.index:
        print(f"Source reference: {row['source_reference']}")

print("\nInspection only. No dataset rows were modified.")
