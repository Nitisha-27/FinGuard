
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data" / "processed" / "finguard_published.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "finguard_grouped.csv"

df = pd.read_csv(INPUT_FILE)

if df["text"].isna().any():
    raise ValueError("Missing text values found in the dataset.")

texts = df["text"].astype(str).tolist()

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    stop_words="english"
)
matrix = vectorizer.fit_transform(texts)

model = NearestNeighbors(metric="cosine", algorithm="brute")
model.fit(matrix)

distances, indices = model.radius_neighbors(
    matrix,
    radius=0.25,
    return_distance=True
)

parent = list(range(len(df)))

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(a, b):
    root_a = find(a)
    root_b = find(b)
    if root_a != root_b:
        parent[root_b] = root_a

for i, (row_distances, row_indices) in enumerate(zip(distances, indices)):
    for distance, j in zip(row_distances, row_indices):
        if i != j and distance <= 0.25:
            union(i, int(j))

roots = [find(i) for i in range(len(df))]
root_ids = {
    root: number
    for number, root in enumerate(sorted(set(roots)), start=1)
}

df["similarity_group"] = [root_ids[root] for root in roots]
df.to_csv(OUTPUT_FILE, index=False)

sizes = df["similarity_group"].value_counts()

print("Grouping complete!")
print("Total examples:", len(df))
print("Number of groups:", df["similarity_group"].nunique())
print("Groups with multiple examples:", int((sizes > 1).sum()))
print("Largest group:", int(sizes.max()))
print("Created:", OUTPUT_FILE)
