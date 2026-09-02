from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def plot_feature(
    df: pd.DataFrame,
    feature: str,
    machine_id: str,
    save: bool = True,
) -> None:
    subset = df[df["machine_id"] == machine_id].copy()

    fig = go.Figure()

    normal = subset[subset["is_anomaly"] == 0]
    fig.add_trace(
        go.Scatter(
            x=normal["timestamp"],
            y=normal[feature],
            mode="lines",
            name="normal",
        )
    )

    anomalies = subset[subset["is_anomaly"] == 1]
    fig.add_trace(
        go.Scatter(
            x=anomalies["timestamp"],
            y=anomalies[feature],
            mode="markers",
            name="anomaly",
        )
    )

    fig.update_layout(
        title=f"{feature} - {machine_id}",
        xaxis_title="time",
        yaxis_title=feature,
        height=500,
    )

    if save:
        import os

        os.makedirs("data/artifacts/plots", exist_ok=True)

        file_path = f"data/artifacts/plots/{machine_id}_{feature}.html"
        fig.write_html(file_path)
        print(f"Saved plot: {file_path}")

    fig.show()

    subset = df[df["machine_id"] == machine_id].copy()

    fig = go.Figure()

    # normal data
    normal = subset[subset["is_anomaly"] == 0]
    fig.add_trace(
        go.Scatter(
            x=normal["timestamp"],
            y=normal[feature],
            mode="lines",
            name="normal",
        )
    )

    # anomalies
    anomalies = subset[subset["is_anomaly"] == 1]
    fig.add_trace(
        go.Scatter(
            x=anomalies["timestamp"],
            y=anomalies[feature],
            mode="markers",
            name="anomaly",
        )
    )

    fig.update_layout(
        title=f"{feature} - {machine_id}",
        xaxis_title="time",
        yaxis_title=feature,
        height=500,
    )

    fig.show()


def main() -> None:
    df = pd.read_csv("data/raw/synthetic_data.csv", parse_dates=["timestamp"])

    print(df["anomaly_type"].value_counts())

    # pick one machine
    machine_id = df["machine_id"].unique()[0]

    # plot a few features
    plot_feature(df, "temperature", machine_id)
    plot_feature(df, "vibration", machine_id)
    plot_feature(df, "rpm", machine_id)


if __name__ == "__main__":
    main()
