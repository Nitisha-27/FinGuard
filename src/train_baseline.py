
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Locate project and dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "finguard_published.csv"
)

# Load data
df = pd.read_csv(DATASET_PATH)

# Basic checks
required_columns = {"text", "label"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"Dataset must contain: {required_columns}")

df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]

X = df["text"]
y = df["label"]

# Split before fitting text features to avoid test-set leakage
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Text classification pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=5000
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])

# Train and evaluate
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Training examples:", len(X_train))
print("Testing examples:", len(X_test))
print("\nAccuracy:", round(accuracy_score(y_test, predictions), 4))
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))
