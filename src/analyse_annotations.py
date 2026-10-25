
from pathlib import Path
import pandas as pd

# 1. Locate and load the annotation dataset
project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "processed" / "annotation_starter.csv"

df = pd.read_csv(data_path)

# 2. Define the annotator columns
annotator_columns = [
    "annotator_1",
    "annotator_2",
    "annotator_3"
]

# 3. Check whether all annotators agree on each example
df["all_agree"] = df[annotator_columns].nunique(axis=1) == 1

# 4. Check whether at least two annotators disagree
df["disagreement"] = ~df["all_agree"]

# 5. Calculate the overall agreement and disagreement rates
total_examples = len(df)
agreement_count = df["all_agree"].sum()
disagreement_count = df["disagreement"].sum()

agreement_rate = agreement_count / total_examples * 100
disagreement_rate = disagreement_count / total_examples * 100

print("Total examples:", total_examples)
print("Examples where all annotators agree:", agreement_count)
print("Examples with annotator disagreement:", disagreement_count)

print(f"\nFull agreement rate: {agreement_rate:.2f}%")
print(f"Disagreement rate: {disagreement_rate:.2f}%")

# 6. Show individual examples and their agreement status
print("\nExample-level results:")
print(
    df[["text"] + annotator_columns + ["all_agree"]]
    .to_string(index=False)
)

# 7. Save the analysis results
output_path = project_root / "data" / "processed" / "annotation_analysis.csv"
df.to_csv(output_path, index=False)

print(f"\nAnalysis saved to: {output_path}")
