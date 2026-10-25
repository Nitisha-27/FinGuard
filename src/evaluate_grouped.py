
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "finguard_grouped.csv"

df = pd.read_csv(DATA_PATH).dropna(subset=["text", "label", "similarity_group"])
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

all_true = []
all_pred = []

print("GROUP-AWARE FIVE-FOLD EVALUATION")
print("================================")

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups=groups), start=1
):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])

    overlap = train_groups.intersection(test_groups)
    if overlap:
        raise RuntimeError(f"Group leakage detected: {overlap}")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    all_true.extend(y_test.tolist())
    all_pred.extend(predictions.tolist())

    print(
        f"Fold {fold}: "
        f"train={len(train_idx)}, test={len(test_idx)}, "
        f"accuracy={accuracy_score(y_test, predictions):.3f}, "
        f"macro-F1={f1_score(y_test, predictions, average='macro', zero_division=0):.3f}"
    )

print("\nOverall out-of-fold results")
print("Accuracy:", round(accuracy_score(all_true, all_pred), 4))
print("Macro-F1:", round(
    f1_score(all_true, all_pred, average="macro", zero_division=0), 4
))
print("\nClassification report:")
print(classification_report(all_true, all_pred, zero_division=0))

print(
    "\nNote: results are exploratory because all 300 candidate examples "
    "still require source-passage verification and label review."
)
