
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "finguard_dataset.csv"

df = pd.read_csv(DATA_PATH)

# Each topic has a financial subject, metric, condition,
# possible action, and outcome.
topics = [
    ("bank deposits", "deposit rates", "market rates change",
     "compare deposit accounts", "interest income"),
    ("government bonds", "bond yields", "inflation expectations change",
     "review bond holdings", "portfolio risk"),
    ("corporate bonds", "credit spreads", "credit conditions tighten",
     "reduce exposure to lower-rated bonds", "credit risk"),
    ("equities", "share prices", "earnings expectations change",
     "increase equity exposure", "potential returns"),
    ("technology stocks", "earnings growth", "technology demand changes",
     "buy technology stocks", "investment performance"),
    ("mutual funds", "expense ratios", "fund fees differ",
     "switch to a lower-cost fund", "investment costs"),
    ("index funds", "tracking error", "market volatility increases",
     "choose an index fund", "tracking performance"),
    ("retirement portfolios", "asset allocation", "retirement approaches",
     "rebalance the portfolio", "long-term risk"),
    ("foreign currencies", "exchange rates", "interest-rate expectations change",
     "increase foreign-currency exposure", "currency returns"),
    ("government securities", "market prices", "market yields rise",
     "sell existing securities", "price risk"),
    ("corporate earnings", "operating profit", "input costs change",
     "invest in the company", "profitability"),
    ("insurance products", "policy charges", "product terms differ",
     "switch insurance products", "total cost"),
    ("exchange-traded funds", "trading volume", "market activity changes",
     "purchase the ETF", "trading liquidity"),
    ("savings accounts", "annual percentage yields", "banks change rates",
     "move savings to another account", "savings income"),
    ("mortgages", "borrowing rates", "lending conditions change",
     "refinance the mortgage", "borrowing costs"),
    ("commodities", "commodity prices", "global supply changes",
     "increase commodity exposure", "price movements"),
    ("pension funds", "funding levels", "market returns fluctuate",
     "change pension investments", "retirement outcomes"),
    ("small-cap stocks", "price volatility", "economic uncertainty increases",
     "reduce small-cap exposure", "portfolio volatility"),
    ("dividend stocks", "dividend yields", "company earnings change",
     "buy dividend-paying shares", "investment income"),
]

# Five patterns per topic = 95 examples per category.
informational_templates = [
    "The {asset} are affected by changes in {metric}.",
    "Changes in {condition} can influence {metric} for {asset}.",
    "The relationship between {metric} and {outcome} depends on market conditions.",
    "Investors use {metric} to assess aspects of {asset}.",
    "Movements in {metric} can affect the risk or return characteristics of {asset}.",
]

advisory_templates = [
    "You should {action} because it may affect {outcome}.",
    "Consider whether you should {action} given that {condition}.",
    "To pursue your stated goal, {action}.",
    "I recommend that you {action} in response to changes in {metric}.",
    "Take action now to {action} before conditions change.",
]

uncertain_templates = [
    "Given the current outlook, {asset} look attractive.",
    "It may be worth considering whether to {action}.",
    "The recent movement in {metric} could make {asset} an opportunity.",
    "A change in {condition} may make it sensible to {action}.",
    "Depending on your objectives, {asset} could be worth a closer look.",
]

# These are provisional template labels, not human-validated ground truth.
categories = [
    ("I", "Informational", informational_templates),
    ("A", "Advisory", advisory_templates),
    ("U", "Uncertain", uncertain_templates),
]

new_rows = []

for code, label, templates in categories:
    for topic_index, topic in enumerate(topics, start=1):
        asset, metric, condition, action, outcome = topic

        for pattern_index, template in enumerate(templates, start=1):
            text = template.format(
                asset=asset,
                metric=metric,
                condition=condition,
                action=action,
                outcome=outcome,
            )

            example_id = (
                f"FG-GEN-{code}-{(topic_index - 1) * 5 + pattern_index:03d}"
            )

            new_rows.append({
                "example_id": example_id,
                "text": text,
                "context": f"General financial discussion involving {asset}.",
                "source_type": "generated",
                "source_reference": "Synthetic template; see src/expand_dataset.py",
                "generation_prompt": (
                    f"Template-based candidate for {label}; "
                    f"topic={asset}; pattern={pattern_index}"
                ),
                "label": label,
                "annotator_confidence": None,
                "linguistic_rationale": (
                    "Provisional template rationale; requires human review."
                ),
                "review_status": "candidate",
            })

# Append only IDs that are not already present.
existing_ids = set(df["example_id"].astype(str))
new_df = pd.DataFrame(new_rows)
new_df = new_df[~new_df["example_id"].isin(existing_ids)]

if not new_df.empty:
    df = pd.concat([df, new_df], ignore_index=True)

# Check dataset integrity before saving.
if df["example_id"].duplicated().any():
    raise ValueError("Duplicate example IDs detected; file was not saved.")

if len(df) != 300:
    raise ValueError(
        f"Expected 300 total rows, found {len(df)}. "
        "Check the existing dataset before continuing."
    )

df.to_csv(DATA_PATH, index=False)

print("FinGuard candidate dataset expanded successfully!")
print("Total rows:", len(df))
print("\nLabel distribution:")
print(df["label"].value_counts())
print("\nSource distribution:")
print(df["source_type"].value_counts())
print("\nReview status:")
print(df["review_status"].value_counts())
print("\nNew rows added this run:", len(new_df))
print("Saved to:", DATA_PATH)
