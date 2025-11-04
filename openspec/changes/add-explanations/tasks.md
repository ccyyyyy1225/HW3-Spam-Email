## Tasks

### Data & Model
- [ ] Ensure vectorizer stores token mapping.
- [ ] Add helper: `explain_linear(sample_text, vectorizer, model, top_k=10)`.

### CLI
- [ ] Add `--explain` flag to `scripts/predict.py`.
- [ ] When set, print token contributions per prediction or save to CSV.

### Streamlit
- [ ] Add checkbox “Show explanations”.
- [ ] Display DataFrame of tokens + weights.
- [ ] Handle batch CSV upload + downloadable results.

### Tests & Docs
- [ ] Add unit test for `explain_linear`.
- [ ] Update README with screenshots and usage.
