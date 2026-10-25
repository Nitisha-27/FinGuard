# FinGuard

**Financial Language Classification Using Large Language Models**

FinGuard is a research project exploring how financial statements can be classified into three categories:

- **Informational:** Provides financial facts or explanations without recommending a specific action.
- **Advisory:** Recommends, encourages, or directs a financial action.
- **Uncertain:** The available wording does not clearly establish whether a statement is informational or advisory.

## Project Objectives

- Build an LLM-based financial language classification pipeline.
- Compare LLM predictions with a traditional machine-learning baseline.
- Investigate ambiguity at the boundary between informational and advisory financial language.
- Evaluate performance using similarity-group-aware validation.
- Develop a research application with a Python backend and React frontend.

## Current Progress

- Created a preliminary dataset containing 300 candidate examples.
- Built a TF-IDF and Logistic Regression baseline.
- Implemented five-fold group-aware evaluation.
- Evaluated the baseline at 80.67% accuracy and 0.8064 macro-F1 on the current candidate dataset.

**Important:** These results are preliminary. The candidate examples still require source-passage verification and label review. They should not be interpreted as validated real-world performance.

## Technology Stack

- Python
- pandas
- scikit-learn
- TF-IDF
- Logistic Regression
- Planned: FastAPI, React, and an LLM API

## Repository Structure

```text
Finguard/
├── data/
│   └── processed/
├── src/
├── requirements.txt
├── .gitignore
└── readme.md
```

## Current Limitations

- The dataset is preliminary and not fully source-verified.
- Similar examples may limit generalization to new financial statements.
- The Uncertain category requires further evaluation.
- No independent human-annotator agreement has been established.

## Future Work

1. Implement and test the LLM classification pipeline.
2. Compare LLM results with the existing baseline.
3. Improve structured outputs and uncertainty handling.
4. Verify source passages and refine labels.
5. Build the FinGuard web application.

---

This project is under active development.
