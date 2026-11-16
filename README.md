# 🌊 FloodGuard AI

**Flood impact & next-day river-level risk intelligence for India** — a Streamlit research prototype built using historical Central Water Commission (CWC)-derived station observations.

## Features
- Dark monitoring dashboard and latest station risk summaries
- Station water-level and rainfall history
- Random Forest next-day Low / Medium / High threshold category predictions
- Separate new-High-risk escalation alerts with a validation-selected **0.60 score cutoff**
- Batch CSV predictions and downloads

## Run locally

1. Install Python 3.10–3.12.
2. Export `risk_model.joblib` and `escalation_model.joblib` from Colab into the `models/` folder (see `COLAB_EXPORT.md`).
3. In the project folder:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

4. Upload a CSV or Parquet file with **at least three consecutive days per station**, with these columns:

`station_code,date_ist,water_level,rainfall,warning_level,danger_level`

Optional: `name`, `lat`, `lon`. Each row should represent one station on one day. Input water levels and station thresholds should be in the same unit (meters), and rainfall should use the same aggregation definition as training.

A small synthetic format example is included at `sample_data/example_format.csv`. **Synthetic values are only for testing the app interface, not for assessing predictive performance.**

## Model evaluation

| Model | Evaluation result |
|---|---|
| Next-day risk Random Forest | 97.91% test accuracy; macro F1 0.85 |
| Persistence baseline | 98.59% test accuracy; macro F1 0.87 |
| Original model, new-High-risk detection | 10.31% (202/1960) |
| Escalation model (previous training configuration) | 51.07% recall, ~19.8% precision at 0.50 on test |
| **Retrained escalation model** | **Validation** at 0.60: precision 24.9%, recall 34.3%, F1 0.288; **held-out test not yet evaluated** |

The original risk classifier **does not outperform the persistence baseline overall**. The escalation classifier is experimental and has substantial false alerts and missed events. Its score is **not a calibrated flood probability**.

## Data provenance

Historical CWC observations aggregated by the third-party [cwc-flood-forecasts project](https://github.com/diagram-chasing/cwc-flood-forecasts). This app does not use live data or independently verified flood impact records. Confirm the semantics of the daily rainfall `sum` field, historical threshold validity, and dataset permissions before publishing derived datasets.

## Safety and limitations

Not for emergency decisions or public flood warnings. A High label means a station-specific *danger water-level threshold exceedance*, not a verified flood disaster. This tool is a portfolio research demonstration and is not affiliated with CWC.

## Project structure

```text
floodguard_ai/
├── app.py
├── requirements.txt
├── COLAB_EXPORT.md
├── README.md
├── models/                 # add your trained .joblib files
└── sample_data/
    └── example_format.csv
```
