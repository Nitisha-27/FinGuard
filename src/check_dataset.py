
from pathlib import Path
import pandas as pd

# Locate the project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load the published-source dataset
DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "finguard_published.csv"
)

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

print("\nFirst five examples:")
print(df[["example_id", "text", "label", "source_reference"]].head())
