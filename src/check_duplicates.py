
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "finguard_published.csv"
)

df = pd.read_csv(DATASET_PATH)
df = df.dropna(subset=["text"]).reset_index(drop=True)

texts = df["text"].astype(str).tolist()

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    stop_words="english"
)

matrix = vectorizer.fit_transform(texts)
similarities = cosine_similarity(matrix)

threshold = 0.75
pairs = []

for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        score = similarities[i, j]

        if score >= threshold:
            pairs.append({
                "similarity": round(float(score), 3),
                "id_1": df.loc[i, "example_id"],
                "label_1": df.loc[i, "label"],
                "text_1": texts[i],
                "id_2": df.loc[j, "example_id"],
                "label_2": df.loc[j, "label"],
                "text_2": texts[j],
            })

results = pd.DataFrame(pairs)

print("Total examples:", len(df))
print("Similarity threshold:", threshold)
print("Near-duplicate pairs found:", len(results))

if not results.empty:
    results = results.sort_values(
        "similarity", ascending=False
    )
    print("\nTop 20 most similar pairs:")
    print(results.head(20).to_string(index=False))

    output_path = (
        PROJECT_ROOT / "data" / "processed"
        / "near_duplicate_pairs.csv"
    )
    results.to_csv(output_path, index=False)
    print("\nAll matching pairs saved to:", output_path)
else:
    print("No pairs exceeded the threshold.")
