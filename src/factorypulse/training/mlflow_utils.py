from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, confusion_matrix


def setup_mlflow(tracking_uri: str, experiment_name: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_params_flat(params: dict[str, Any], prefix: str | None = None) -> None:
    for key, value in params.items():
        full_key = f"{prefix}.{key}" if prefix else key
        mlflow.log_param(full_key, value)


def log_metrics_with_prefix(metrics: dict[str, float], prefix: str) -> None:
    for key, value in metrics.items():
        mlflow.log_metric(f"{prefix}_{key}", float(value))


def log_artifact_if_exists(path: str | Path) -> None:
    path = Path(path)
    if path.exists():
        mlflow.log_artifact(str(path))


def log_text_artifact(content: str, artifact_file: str) -> None:
    mlflow.log_text(content, artifact_file)


def save_confusion_matrix_plot(
    y_true: pd.Series,
    y_pred: pd.Series,
    output_path: str | Path,
    title: str,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def save_pr_curve_plot(
    y_true: pd.Series,
    y_score: pd.Series,
    output_path: str | Path,
    title: str,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax)
    disp.ax_.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def save_timeseries_anomaly_plot(
    df: pd.DataFrame,
    timestamp_column: str,
    value_column: str,
    true_label_column: str,
    pred_label_column: str,
    output_path: str | Path,
    title: str,
    max_points: int = 1500,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = df.copy().sort_values(timestamp_column)

    if len(plot_df) > max_points:
        step = max(1, len(plot_df) // max_points)
        plot_df = plot_df.iloc[::step].copy()

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(plot_df[timestamp_column], plot_df[value_column], label=value_column, linewidth=1.2)

    true_anoms = plot_df[plot_df[true_label_column] == 1]
    pred_anoms = plot_df[plot_df[pred_label_column] == 1]

    ax.scatter(
        true_anoms[timestamp_column],
        true_anoms[value_column],
        label="true_anomaly",
        marker="o",
        s=20,
    )
    ax.scatter(
        pred_anoms[timestamp_column],
        pred_anoms[value_column],
        label="predicted_anomaly",
        marker="x",
        s=28,
    )

    ax.set_title(title)
    ax.set_xlabel("timestamp")
    ax.set_ylabel(value_column)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def save_metrics_summary_csv(
    rows: list[dict[str, object]],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    return output_path