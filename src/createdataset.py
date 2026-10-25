
from pathlib import Path
import pandas as pd

# Locate the FinGuard project
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Define the original FinGuard dataset structure
columns = [
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

# Create an empty dataset with the required columns
df = pd.DataFrame(columns=columns)

output_path = DATA_DIR / "finguard_dataset.csv"

# Do not overwrite an existing dataset
if output_path.exists():
    print(f"Dataset already exists: {output_path}")
    print("No files were overwritten.")
else:
    df.to_csv(output_path, index=False)
    print("FinGuard dataset template created!")
    print(f"Location: {output_path}")
    print("\nColumns:")
    for column in columns:
        print("-", column)
    print("\nInitial number of examples:", len(df))
