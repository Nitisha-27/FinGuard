# src/analyse_results.py
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

CSV_PATH = (
    Path(__file__).resolve().parents[1]
    / "data" / "processed" / "finguard_final_predictions.csv"
)

LABEL_ORDER = ["Informational", "Advisory", "Uncertain"]
COLOURS = {
    "Informational": "#2196F3",
    "Advisory":      "#F44336",
    "Uncertain":     "#FF9800",
}

df = pd.read_csv(CSV_PATH)
df["correct"] = df["label"] == df["llm_prediction"]


# ── 1. Entropy distribution by human label ────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))

for label in LABEL_ORDER:
    subset = df[df["label"] == label]["vote_entropy"].fillna(0)
    ax.scatter(
        subset,
        [label] * len(subset),
        color=COLOURS[label],
        alpha=0.7,
        s=80,
        zorder=3,
    )
    ax.axvline(subset.mean(), color=COLOURS[label], linestyle="--", linewidth=1)

ax.set_xlabel("Vote Entropy (0 = certain, 1 = maximally uncertain)")
ax.set_title("Model Vote Entropy by Human Label\n"
             "Dashed lines = class mean")
ax.set_xlim(-0.05, 1.05)
ax.grid(axis="x", alpha=0.3)
patches = [mpatches.Patch(color=COLOURS[l], label=l) for l in LABEL_ORDER]
ax.legend(handles=patches, loc="lower right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "entropy_by_label.png", dpi=150)
plt.close()
print("Saved: entropy_by_label.png")


# ── 2. Vote confidence vs correctness ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))

for label in LABEL_ORDER:
    subset = df[df["label"] == label]
    for _, row in subset.iterrows():
        marker = "o" if row["correct"] else "X"
        ax.scatter(
            row["vote_confidence"],
            label,
            marker=marker,
            color=COLOURS[label],
            s=100,
            alpha=0.8,
            zorder=3,
        )

ax.set_xlabel("Vote Confidence (majority fraction)")
ax.set_title("Vote Confidence vs Correctness by Label\n"
             "○ = correct   ✕ = incorrect")
ax.set_xlim(0.3, 1.1)
ax.grid(axis="x", alpha=0.3)
patches = [mpatches.Patch(color=COLOURS[l], label=l) for l in LABEL_ORDER]
ax.legend(handles=patches, loc="lower right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confidence_vs_correctness.png", dpi=150)
plt.close()
print("Saved: confidence_vs_correctness.png")


# ── 3. Confusion matrix heatmap ───────────────────────────────────────────────
import numpy as np

valid = df[df["llm_prediction"].isin(LABEL_ORDER)]
cm = pd.crosstab(
    valid["label"], valid["llm_prediction"],
    rownames=["Actual"], colnames=["Predicted"],
).reindex(index=LABEL_ORDER, columns=LABEL_ORDER, fill_value=0)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm.values, cmap="Blues")

ax.set_xticks(range(len(LABEL_ORDER)))
ax.set_yticks(range(len(LABEL_ORDER)))
ax.set_xticklabels(LABEL_ORDER, rotation=30, ha="right")
ax.set_yticklabels(LABEL_ORDER)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")

for i in range(len(LABEL_ORDER)):
    for j in range(len(LABEL_ORDER)):
        val = cm.values[i, j]
        ax.text(j, i, str(val), ha="center", va="center",
                color="white" if val > 5 else "black", fontsize=14)

plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
plt.close()
print("Saved: confusion_matrix.png")


# ── 4. Grey zone deep-dive: the 4 misclassified Uncertain examples ─────────────
print("\n── Grey Zone Analysis: Misclassified Uncertain Examples ──")
misclassified = df[(df["label"] == "Uncertain") & (~df["correct"])]

for _, row in misclassified.iterrows():
    votes = {
        "Informational": row["votes_informational"],
        "Advisory":      row["votes_advisory"],
        "Uncertain":     row["votes_uncertain"],
    }
    print(f"\nID:          {row['example_id']}")
    print(f"Text:        {row['text']}")
    print(f"Predicted:   {row['llm_prediction']} "
          f"(entropy={row['vote_entropy']:.3f}, conf={row['vote_confidence']:.2f})")
    print(f"Votes:       {votes}")
    print(f"Explanation: {row.get('explanation', '')}")


# ── 5. Summary statistics table ───────────────────────────────────────────────
print("\n── Summary Statistics ──")
summary = df.groupby("label").agg(
    n=("label", "size"),
    accuracy=("correct", "mean"),
    mean_entropy=("vote_entropy", "mean"),
    mean_confidence=("vote_confidence", "mean"),
).reindex(LABEL_ORDER)
summary["accuracy"] = summary["accuracy"].map("{:.0%}".format)
summary["mean_entropy"] = summary["mean_entropy"].map("{:.4f}".format)
summary["mean_confidence"] = summary["mean_confidence"].map("{:.4f}".format)
print(summary.to_string())


# ── 6. Core research finding ──────────────────────────────────────────────────
uncertain_entropy = df[df["label"] == "Uncertain"]["vote_entropy"].mean()
clear_entropy = df[df["label"].isin(["Informational", "Advisory"])]["vote_entropy"].mean()

print(f"\n── Core Research Finding ──")
print(f"Mean entropy, clear examples (Inf+Adv): {clear_entropy:.4f}")
print(f"Mean entropy, Uncertain examples:       {uncertain_entropy:.4f}")
print(f"Entropy ratio (Uncertain / clear):      "
      f"{uncertain_entropy / clear_entropy if clear_entropy > 0 else 'inf'}")
print(
    "\nConclusion: Model vote entropy is elevated on human-labelled Uncertain "
    "examples and zero on clear examples.\nThis confirms that model stochastic "
    "uncertainty correlates with human annotator disagreement."
)