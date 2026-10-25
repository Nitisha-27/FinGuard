
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "finguard_grouped.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "grouped_oof_predictions.csv"

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["text", "label", "similarity_group"]).copy()
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""].reset_index(drop=True)

X = df["text"]
y = df["label"]
groups = df["similarity_group"]

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=5000
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

predictions = [None] * len(df)

for train_idx, test_idx in cv.split(X, y, groups=groups):
    model.fit(X.iloc[train_idx], y.iloc[train_idx])

    fold_predictions = model.predict(X.iloc[test_idx])

    for idx, prediction in zip(test_idx, fold_predictions):
        predictions[idx] = prediction

df["predicted_label"] = predictions
df["is_correct"] = df["label"] == df["predicted_label"]

# Keep the original text and identifiers so mistakes can be reviewed.
df.to_csv(OUTPUT_PATH, index=False)

labels = sorted(df["label"].unique())
matrix = confusion_matrix(
    df["label"],
    df["predicted_label"],
    labels=labels
)

print("CONFUSION MATRIX")
print("Rows = actual labels; columns = predicted labels")
print("Label order:", labels)
print(matrix)

print("\nERROR COUNT")
print("Incorrect predictions:", int((~df["is_correct"]).sum()))
print("Correct predictions:", int(df["is_correct"].sum()))

print("\nERRORS BY ACTUAL LABEL")
errors = df[~df["is_correct"]].copy()
print(errors["label"].value_counts().to_string())

print("\nSAVED MISCLASSIFICATIONS")
print(OUTPUT_PATH)
print("Total errors:", len(errors))

print("\nFIRST 15 ERRORS")
for _, row in errors.head(15).iterrows():
    print("\nID:", row["example_id"])
    print("Actual:", row["label"])
    print("Predicted:", row["predicted_label"])
    print("Text:", row["text"])
