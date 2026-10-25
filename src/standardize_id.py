
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "processed" / "finguard_dataset.csv"

df = pd.read_csv(data_path)

# Keep a backup before changing IDs
backup_path = data_path.with_name("finguard_dataset_backup.csv")

if not backup_path.exists():
    df.to_csv(backup_path, index=False)
    print("Backup created:", backup_path)

# Assign a single, consistent ID format
df["example_id"] = [
    f"FG-{number:04d}"
    for number in range(1, len(df) + 1)
]

# Verify uniqueness
assert df["example_id"].is_unique

df.to_csv(data_path, index=False)

print("IDs standardized successfully.")
print("Total rows:", len(df))
print("First five IDs:", df["example_id"].head().tolist())
print("Last five IDs:", df["example_id"].tail().tolist())
print("Unique IDs:", df["example_id"].nunique())
