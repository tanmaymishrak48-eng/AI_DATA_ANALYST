# 🤖 AI Data Analyst

**A Generative AI-Powered Platform for Natural-Language Data Analysis, Visualization and Business Insights**

### 🚀 Live Demo

👉 https://ai-data-analyst-tanmaya.streamlit.app/

AI Data Analyst is a Streamlit web application that lets a non-technical user upload a CSV/Excel dataset and then explore it the way they'd talk to a human analyst — by asking questions in plain English, reading automatically generated business recommendations, and exporting everything as a PDF report.

---

## What it does

| Page | Capability |
|---|---|
| 📊 **Dashboard** | Row/column counts, missing values, duplicates, and key business KPIs (Revenue, Profit, Quantity) computed instantly from the uploaded file. |
| 📈 **Analytics** | Visual exploration of the dataset (distributions, breakdowns) via Plotly. |
| 💬 **Ask Data** | Natural-language Q&A over the dataset — e.g. *"Which region has the highest revenue?"* |
| 🤖 **AI Insights** | Groq-generated plain-English summary of the dataset's structure and quality. |
| 🚨 **Anomalies** | Statistical (IQR-based) outlier detection, with an optional AI explanation of *why* the flagged values might matter. |
| 💡 **Recommendations** | Evidence-based business recommendations generated from verified dataset statistics. |
| 📄 **Reports** | One-click PDF export bundling every section above. |

---

## Architecture — how the "generative AI" part actually works

The single most important design decision in this project is **keeping the LLM out of the arithmetic**:

```
Natural-language question
        │
        ▼
  Groq (ai_engine.py)  ──►  structured JSON plan
        │                      { "operation": "sum_by_group",
        │                        "column": "Revenue",
        │                        "group_by": "Region" }
        ▼
  Pandas (data_analyzer.py)  ──►  the real, verified number
        │
        ▼
  Result shown to the user
```

Groq only ever decides **what** the user is asking for (which operation, which column, which grouping). The actual sum/mean/count is always computed by pandas — Groq never invents or "hallucinates" a number. The same pattern is used for the AI Insights, Anomaly Explanation, and Business Recommendations pages: pandas computes the evidence (totals, group-by breakdowns, anomaly counts) as a JSON object, and Groq's prompt is explicitly instructed to interpret that evidence rather than invent new statistics.

This trade-off does mean the natural-language interface only supports a fixed set of operations (`sum`, `mean`, `count`, and their grouped variants, plus highest/lowest group). More open-ended questions (correlations, multi-column filters, forecasting) are a natural next step, deliberately left out of this version in favor of correctness and safety.

### Anomaly detection
Outliers are flagged using the **IQR method**: for each numeric column, any value below `Q1 − 1.5×IQR` or above `Q3 + 1.5×IQR` is flagged, along with the column, the bounds, and the direction (unusually high/low). This is a standard, explainable statistical technique — not an LLM guess.

---

## Project structure

```
AI_DATA_ANALYST/
├── app.py                 # Streamlit UI — all 7 pages
├── ai_engine.py            # Groq prompts + JSON-plan generation + retry/quota handling
├── data_analyzer.py         # Pure pandas calculation layer (executes the AI's plan)
├── anomaly_detector.py       # IQR-based outlier detection
├── report_generator.py        # PDF export (ReportLab)
├── ai_data_analyst_sample_sales.csv   # Sample dataset for trying the app / running tests
├── requirements.txt
├── .env.example             # Copy to .env and add your own Groq API key
├── .gitignore
└── tests/
    ├── conftest.py                     # shared fixtures + a `requires_groq` marker
    ├── test_data_analyzer.py            # offline, no API needed
    ├── test_anomaly_detector.py          # offline, no API needed
    └── test_ai_engine_integration.py      # calls live Groq — skipped without a key
```

---

## Setup

1. **Clone/unzip the project, then create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your Groq API key:**
   ```bash
   cp .env.example .env
   # then edit .env and paste your key:
   # GROQ_API_KEY=your_actual_key
   ```
   Get a free key at https://console.groq.com/keys. By default the app uses Groq's `openai/gpt-oss-120b` model (set in `ai_engine.py`) — a currently free-tier production model on Groq. If you'd rather use a faster/lighter option, swap in `openai/gpt-oss-20b`; check https://console.groq.com/docs/models for the full current list, since Groq periodically moves models between free and Enterprise-only tiers.

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

5. **Try it immediately** with the bundled `ai_data_analyst_sample_sales.csv` — it has `Revenue`, `Profit`, `Quantity`, `Region`, `Category`, and `Discount` columns, which every dashboard KPI and business-recommendation rule looks for.

---

## Running the tests

```bash
pip install pytest
pytest tests/ -v
```

- `test_data_analyzer.py` and `test_anomaly_detector.py` run **offline** — no API key needed — and check the calculation and anomaly-detection logic directly against known-correct pandas results.
- `test_ai_engine_integration.py` exercises the live Groq pipeline end to end (question → plan → pandas execution). These are **skipped automatically** if `GROQ_API_KEY` isn't set, so the suite still passes cleanly in an environment with no key or network access.

---

## Known limitations / next steps

- The natural-language operation set is fixed (sum/mean/count and grouped variants). Extending it to correlations, filters, and multi-step questions is the natural next phase.
- Business-metric detection (Revenue/Profit/Quantity/Region/Category) currently looks for specific column names — a future version could let the user map their own column names, or have Groq infer the mapping.
- Anomaly detection currently covers numeric columns only (IQR method); categorical/rare-category anomalies are not yet detected.

---

## Tech stack

Streamlit · Pandas · NumPy · Plotly · Groq API (`groq`) · ReportLab (PDF export) · pytest
