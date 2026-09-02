from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.utils import (  # noqa: E402
    build_base_dataset,
    load_prediction_file,
    plot_time_series_with_anomalies,
)

st.title("Anomaly Visualization")

base_df = build_base_dataset()

if base_df.empty:
    st.warning("No processed datasets found.")
    st.stop()

model_to_file = {
    "Rolling Z-score": "rolling_zscore_test_predictions.csv",
    "Isolation Forest": "isolation_forest_test_predictions.csv",
    "LSTM Autoencoder": "lstm_autoencoder_test_predictions.csv",
}

model_option = st.selectbox("Select model predictions", list(model_to_file.keys()))
pred_df = load_prediction_file(model_to_file[model_option])

if pred_df is None:
    st.warning(f"Prediction file not found: {model_to_file[model_option]}")
    st.stop()

signal_column = st.selectbox(
    "Signal",
    ["temperature", "vibration", "pressure", "rpm", "power_draw", "throughput"],
    index=0,
)

machine_id = st.selectbox("Machine", sorted(pred_df["machine_id"].unique()))

base_signal_df = (
    base_df[["machine_id", "timestamp", signal_column]]
    .copy()
    .rename(columns={signal_column: "signal_value"})
)

merged = pred_df.merge(
    base_signal_df,
    on=["machine_id", "timestamp"],
    how="left",
)

merged = merged[merged["machine_id"] == machine_id].copy().sort_values("timestamp")

st.subheader(f"{model_option} — {machine_id}")
plot_time_series_with_anomalies(
    merged,
    value_column="signal_value",
    title=f"{model_option} - {machine_id} - {signal_column}",
)

st.subheader("Prediction preview")
st.dataframe(merged.head(30), use_container_width=True)
