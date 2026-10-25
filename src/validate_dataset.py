
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "finguard_dataset.csv"

df = pd.read_csv(DATA_PATH)

required_columns = [
    "example_id",
    "text",
    "context",
    "source_type",
    "source_reference",
    "generation_prompt",
    "label",
    "annotator_confidence",
    "linguistic_rationale",
    "review_status",
]

allowed_labels = {"Informational", "Advisory", "Uncertain"}
allowed_sources = {"collected", "generated"}

errors = []

# Check the schema
missing_columns = set(required_columns) - set(df.columns)
if missing_columns:
    errors.append(f"Missing columns: {sorted(missing_columns)}")

if not missing_columns:
    # Check unique IDs
    if df["example_id"].isna().any():
        errors.append("Some example IDs are missing.")

    if df["example_id"].duplicated().any():
        errors.append("Duplicate example IDs found.")

    # Check required text fields
    for column in ["text", "source_type", "label"]:
        if df[column].isna().any() or df[column].astype(str).str.strip().eq("").any():
            errors.append(f"Missing or empty values in '{column}'.")

    # Check labels and source types
    invalid_labels = set(df["label"].dropna()) - allowed_labels
    if invalid_labels:
        errors.append(f"Invalid labels: {sorted(invalid_labels)}")

    invalid_sources = set(df["source_type"].dropna()) - allowed_sources
    if invalid_sources:
        errors.append(f"Invalid source types: {sorted(invalid_sources)}")

    # Check confidence values
    confidence = pd.to_numeric(df["annotator_confidence"], errors="coerce")
    if confidence.isna().any():
        errors.append("Some confidence values are missing or not numeric.")
    elif not confidence.between(0, 1).all():
        errors.append("Confidence values must be between 0 and 1.")

    # Check source provenance
    if df["source_reference"].isna().any():
        errors.append("Some source references are missing.")

# Report the results
print("Dataset rows:", len(df))

if errors:
    print("\nValidation FAILED:")
    for error in errors:
        print("-", error)
else:
    print("Validation PASSED!")

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    print("\nSource distribution:")
    print(df["source_type"].value_counts())

    print("\nDuplicate IDs:", df["example_id"].duplicated().sum())
