from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.utils import (  # noqa: E402
    compute_metrics_from_predictions,
    load_prediction_file,
    plot_confusion_matrix,
    plot_metric_bars,
)

st.title("Model Performance")

files = {
    "Rolling Z-score": "rolling_zscore_test_predictions.csv",
    "Isolation Forest": "isolation_forest_test_predictions.csv",
    "LSTM Autoencoder": "lstm_autoencoder_test_predictions.csv",
}

metric_rows: list[dict[str, float | str]] = []

for model_name, filename in files.items():
    pred_df = load_prediction_file(filename)
    if pred_df is None:
        continue

    metrics = compute_metrics_from_predictions(pred_df)
    metric_rows.append({"model": model_name, **metrics})

if not metric_rows:
    st.warning("No prediction files found for performance comparison.")
    st.stop()

metrics_df = pd.DataFrame(metric_rows)

st.subheader("Metrics table")
st.dataframe(metrics_df, use_container_width=True)

st.subheader("Metric comparison")
plot_metric_bars(metrics_df)

selected_model = st.selectbox("Show confusion matrix for model", metrics_df["model"].tolist())
selected_pred_df = load_prediction_file(files[selected_model])

if selected_pred_df is not None:
    st.subheader(f"Confusion Matrix — {selected_model}")
    plot_confusion_matrix(selected_pred_df, f"{selected_model} - Test Confusion Matrix")
