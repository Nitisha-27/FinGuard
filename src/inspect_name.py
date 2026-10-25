
from pathlib import Path
import pandas as pd

# Find the CSV file
project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "raw" / "financial_phrasebank.csv"

# Load the dataset
df = pd.read_csv(data_path)

# Inspect the dataset
print("Dataset dimensions:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate sentences:", df["sentence"].duplicated().sum())

print("\nSentiment counts:")
print(df["sentiment"].value_counts())

print("\nRandom sample of 5 sentences:")
print(df.sample(5, random_state=42).to_string(index=False))
