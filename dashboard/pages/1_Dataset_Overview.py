from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.utils import build_base_dataset, plot_distribution, plot_time_series_with_anomalies


st.title("Dataset Overview")

df = build_base_dataset()

if df.empty:
    st.warning("No processed datasets found in data/processed.")
    st.stop()

st.subheader("Dataset summary")
st.write(
    {
        "rows": len(df),
        "machines": df["machine_id"].nunique(),
        "time_start": str(df["timestamp"].min()),
        "time_end": str(df["timestamp"].max()),
        "anomaly_ratio": float(df["is_anomaly"].mean()),
    }
)

machine_id = st.selectbox("Select machine", sorted(df["machine_id"].unique()))
value_column = st.selectbox(
    "Select signal",
    ["temperature", "vibration", "pressure", "rpm", "power_draw", "throughput"],
    index=0,
)

machine_df = df[df["machine_id"] == machine_id].copy().sort_values("timestamp")
machine_df["prediction"] = 0

plot_time_series_with_anomalies(
    machine_df,
    value_column=value_column,
    title=f"{value_column} - {machine_id}",
)

st.subheader("Feature distribution")
plot_distribution(machine_df, value_column)

st.subheader("Data preview")
st.dataframe(machine_df.head(20), use_container_width=True)