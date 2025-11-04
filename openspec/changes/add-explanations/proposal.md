# Proposal: Add Explainable Predictions

## Why
Users need to understand why an email is classified as spam or ham.  
This improves transparency, trust, and debugging.

## What
- Add explainability layer for model predictions.
- CLI: `--explain` flag outputs token-level contributions.
- Streamlit: checkbox “Show explanations” to display top weighted tokens.
- Use LinearSVC / LogisticRegression coefficients (or SHAP fallback).
- Export explanations to `predictions.csv` if batch mode.

## Success Criteria
- Each prediction displays top 10 contributing tokens.
- Works for both single-text and batch CSV mode.
- Falls back gracefully for unsupported models.
