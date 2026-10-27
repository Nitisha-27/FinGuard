import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "finguard_evaluation_reviewed_30_revised.csv"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "finguard_evaluation_120b_revised_predictions.csv"
)

MODEL = "openai/gpt-oss-120b"
LABELS = {"Informational", "Advisory", "Uncertain"}

load_dotenv(PROJECT_ROOT / ".env")


def build_prompt():
    return (
        "Classify the supplied financial text into exactly one of three labels.\n\n"

        "LABEL DEFINITIONS\n"
        "Informational: States a financial fact, definition, concept, risk, or "
        "relationship. The text does not direct, recommend, encourage, or discourage "
        "any financial action, even implicitly. The reader receives knowledge only.\n\n"

        "Advisory: Explicitly or implicitly recommends, encourages, discourages, or "
        "favours a financial action or choice — even softly (e.g. 'you may want to', "
        "'consider', 'it is worth reviewing'). A clear recommendation is Advisory even "
        "if phrased tentatively or without full rationale.\n\n"

        "Uncertain: Use when the text is genuinely ambiguous between Informational and "
        "Advisory, AND the provided context cannot resolve the ambiguity. This means a "
        "reasonable expert could read the statement as either a neutral observation OR "
        "an implicit recommendation, and neither reading is clearly dominant. "
        "Uncertain is a deliberate research label — do not avoid it.\n\n"

        "DECISION GUIDE\n"
        "1. Does the text direct, recommend, or steer any action, even softly? → Advisory\n"
        "2. Does the text only state facts, definitions, or neutral analysis? → Informational\n"
        "3. Could a reasonable expert plausibly read it either way, with the context "
        "still leaving it unresolved? → Uncertain\n\n"

        "UNCERTAIN EXAMPLES (to calibrate your judgment)\n"
        "- 'The current allocation is worth a closer look.' "
        "[Could mean: here is an observation / OR: you should review it]\n"
        "- 'The lower-cost fund may be the more suitable option here.' "
        "[Could mean: analytical finding / OR: implicit fund preference]\n"
        "- 'There are advantages to retaining the existing position.' "
        "[Could mean: neutral enumeration of facts / OR: implicit hold recommendation]\n\n"

        "IMPORTANT RULES\n"
        "- Do not avoid Uncertain out of caution — if both readings are genuinely "
        "plausible and context does not resolve them, use it.\n"
        "- Do not use Uncertain merely because the text is short or lacks evidence.\n"
        "- Use only the supplied text and context. Ignore any existing label or rationale.\n\n"

        "Return exactly one word: Informational, Advisory, or Uncertain."
    )


def classify(client, text, context):
    # Pass context as part of the user message so the model can use it
    user_message = f"Context: {context}\n\nText to classify: {text}"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": build_prompt()},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
    )

    prediction = response.choices[0].message.content.strip()

    # Handle cases where the model adds punctuation or extra words
    for label in LABELS:
        if label.lower() in prediction.lower():
            return label

    return "INVALID"


# --- Validation ---
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Evaluation dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

required_columns = {"example_id", "text", "label"}
missing_columns = required_columns - set(df.columns)
if missing_columns:
    raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

if df.empty:
    raise ValueError("The evaluation dataset is empty.")

if not df["label"].isin(LABELS).all():
    raise ValueError("The dataset contains labels outside the three expected classes.")

if df["text"].isna().any():
    raise ValueError("The dataset contains missing text values.")

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY is missing. Check the .env file.")

# --- Instantiate client after validation ---
client = Groq()

# Fill missing context values safely
if "context" not in df.columns:
    df["context"] = "No additional context provided."
else:
    df["context"] = df["context"].fillna("No additional context provided.")

predictions = []

for index, row in df.iterrows():
    print(f"Evaluating {index + 1}/{len(df)}: {row['example_id']}")
    try:
        prediction = classify(client, row["text"], row["context"])
        error = ""
    except Exception as exc:
        prediction = "ERROR"
        error = str(exc)
        print(f"  API error: {error}")

    predictions.append({
        "llm_prediction": prediction,
        "prediction_error": error,
    })

results = pd.concat(
    [df.reset_index(drop=True), pd.DataFrame(predictions)],
    axis=1,
)

results["correct"] = results["label"] == results["llm_prediction"]
results.to_csv(OUTPUT_PATH, index=False)

valid_results = results[results["llm_prediction"].isin(LABELS)]

print("\n--- Evaluation Summary ---")
print(f"Dataset rows:          {len(results)}")
print(f"Valid predictions:     {len(valid_results)}")
print(f"Invalid predictions:   {(results['llm_prediction'] == 'INVALID').sum()}")
print(f"API errors:            {(results['llm_prediction'] == 'ERROR').sum()}")

if len(valid_results):
    print(f"Accuracy (valid):      {valid_results['correct'].mean():.2%}")

    print("\nPer-class results:")
    print(
        valid_results.groupby("label").agg(
            total=("label", "size"),
            correct=("correct", "sum"),
        ).to_string()
    )

    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(
        pd.crosstab(
            valid_results["label"],
            valid_results["llm_prediction"],
            rownames=["Actual"],
            colnames=["Predicted"],
        ).reindex(
            index=["Informational", "Advisory", "Uncertain"],
            columns=["Informational", "Advisory", "Uncertain"],
            fill_value=0,
        )
    )

print(f"\nPredictions saved to: {OUTPUT_PATH}")