from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.utils import (  # noqa: E402
    MODEL_DIR,
    PROCESSED_DIR,
    plot_time_series_with_anomalies,
    run_lstm_inference,
)

st.title("Inference Demo")

st.info(
    "This demo expects a processed feature-format CSV "
    "(same schema as train_features/val_features/test_features). "
    "You can also run inference on a sampled segment from the existing test set."
)

if not MODEL_DIR.exists():
    st.warning("Model directory not found: data/artifacts/models")
    st.stop()

mode = st.radio(
    "Inference source",
    ["Use sample from test_features.csv", "Upload processed feature CSV"],
    horizontal=True,
)

input_df: pd.DataFrame | None = None

if mode == "Use sample from test_features.csv":
    test_path = PROCESSED_DIR / "test_features.csv"

    if not test_path.exists():
        st.warning("Missing data/processed/test_features.csv")
        st.stop()

    test_df = pd.read_csv(test_path, parse_dates=["timestamp"])
    machine_ids = sorted(test_df["machine_id"].unique())
    machine_id = st.selectbox("Machine for sample inference", machine_ids)

    machine_df = test_df[test_df["machine_id"] == machine_id].copy().sort_values("timestamp")
    max_rows = len(machine_df)

    sample_size = st.slider(
        "Sample size",
        min_value=100,
        max_value=max(100, min(max_rows, 2000)),
        value=min(max_rows, 500),
        step=50,
    )

    input_df = machine_df.head(sample_size).copy()

else:
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file, parse_dates=["timestamp"])

if input_df is None:
    st.stop()

st.subheader("Input preview")
st.dataframe(input_df.head(20), use_container_width=True)

try:
    pred_df = run_lstm_inference(input_df)
except Exception as exc:
    st.error(f"Inference failed: {exc}")
    st.stop()

if pred_df.empty:
    st.warning("No sequences could be formed. The input may be shorter than the sequence length.")
    st.stop()

st.subheader("Inference summary")
st.write(
    {
        "rows_in_input": len(input_df),
        "sequences_scored": len(pred_df),
        "predicted_anomalies": int(pred_df["prediction"].sum()),
        "mean_anomaly_score": float(pred_df["anomaly_score"].mean()),
        "max_anomaly_score": float(pred_df["anomaly_score"].max()),
    }
)

available_signals = [
    c
    for c in ["temperature", "vibration", "pressure", "rpm", "power_draw", "throughput"]
    if c in input_df.columns
]
signal_column = st.selectbox("Signal for visualization", available_signals, index=0)

merged = pred_df.merge(
    input_df[["machine_id", "timestamp", signal_column]].copy(),
    on=["machine_id", "timestamp"],
    how="left",
).sort_values(["machine_id", "timestamp"])

plot_time_series_with_anomalies(
    merged,
    value_column=signal_column,
    title=f"LSTM Inference Demo - {signal_column}",
)

st.subheader("Predictions")
st.dataframe(pred_df.head(30), use_container_width=True)

csv_bytes = pred_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download predictions CSV",
    data=csv_bytes,
    file_name="lstm_inference_predictions.csv",
    mime="text/csv",
)
