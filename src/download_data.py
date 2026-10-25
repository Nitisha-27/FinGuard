
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile
import pandas as pd

# Locate project folders
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Download the original dataset archive
url = (
    "https://huggingface.co/datasets/"
    "takala/financial_phrasebank/resolve/main/"
    "data/FinancialPhraseBank-v1.0.zip"
)

zip_path = RAW_DIR / "financial_phrasebank.zip"
extract_dir = RAW_DIR / "extracted"

if not zip_path.exists():
    print("Downloading Financial PhraseBank...")
    urlretrieve(url, zip_path)
    print("Download complete!")

# Extract the archive
extract_dir.mkdir(parents=True, exist_ok=True)

with ZipFile(zip_path, "r") as archive:
    archive.extractall(extract_dir)

# Locate the highest-agreement text file
matches = list(
    extract_dir.rglob("Sentences_AllAgree.txt")
)

if not matches:
    raise FileNotFoundError(
        "Could not find Sentences_AllAgree.txt "
        "inside the downloaded archive."
    )

# Parse each sentence and its original sentiment label
records = []

with open(matches[0], "r", encoding="iso-8859-1") as file:
    for line in file:
        line = line.strip()

        if not line:
            continue

        sentence, label = line.rsplit("@", 1)

        records.append({
            "sentence": sentence.strip(),
            "sentiment": label.strip().lower()
        })

df = pd.DataFrame(records)

# Basic validation
assert not df.empty
assert df["sentence"].notna().all()
assert df["sentiment"].isin(
    ["positive", "negative", "neutral"]
).all()

# Save the loaded dataset
output_path = RAW_DIR / "financial_phrasebank.csv"
df.to_csv(output_path, index=False)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nFirst five rows:")
print(df.head().to_string(index=False))
print("\nSentiment distribution:")
print(df["sentiment"].value_counts())
print(f"\nSaved to: {output_path}")
