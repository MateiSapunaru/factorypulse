from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from factorypulse.inference.lstm_inference import load_lstm_bundle, predict_with_lstm_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = DATA_DIR / "artifacts"
PREDICTION_DIR = ARTIFACT_DIR / "predictions"
MODEL_DIR = ARTIFACT_DIR / "models"


@st.cache_data
def load_processed_datasets() -> dict[str, pd.DataFrame]:
    datasets: dict[str, pd.DataFrame] = {}

    for name in ["train_features", "val_features", "test_features"]:
        path = PROCESSED_DIR / f"{name}.csv"
        if path.exists():
            datasets[name] = pd.read_csv(path, parse_dates=["timestamp"])

    return datasets


@st.cache_data
def build_base_dataset() -> pd.DataFrame:
    datasets = load_processed_datasets()
    frames = list(datasets.values())

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    return df


@st.cache_data
def load_prediction_file(filename: str) -> pd.DataFrame | None:
    path = PREDICTION_DIR / filename
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["timestamp"])


def compute_metrics_from_predictions(pred_df: pd.DataFrame) -> dict[str, float]:
    y_true = pred_df["is_anomaly"]
    y_pred = pred_df["prediction"]
    y_score = pred_df["anomaly_score"]

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_true, y_score)

    false_positives = int(((y_true == 0) & (y_pred == 1)).sum())
    fp_per_1000 = false_positives / len(pred_df) * 1000.0 if len(pred_df) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "pr_auc": pr_auc,
        "false_positives_per_1000": fp_per_1000,
    }


def plot_time_series_with_anomalies(
    df: pd.DataFrame,
    value_column: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(df["timestamp"], df[value_column], linewidth=1.2, label=value_column)

    true_anoms = df[df["is_anomaly"] == 1]
    pred_anoms = df[df["prediction"] == 1] if "prediction" in df.columns else pd.DataFrame()

    if not true_anoms.empty:
        ax.scatter(
            true_anoms["timestamp"],
            true_anoms[value_column],
            s=24,
            label="true_anomaly",
        )

    if not pred_anoms.empty:
        ax.scatter(
            pred_anoms["timestamp"],
            pred_anoms[value_column],
            s=28,
            marker="x",
            label="predicted_anomaly",
        )

    ax.set_title(title)
    ax.set_xlabel("timestamp")
    ax.set_ylabel(value_column)
    ax.legend()
    fig.autofmt_xdate()

    st.pyplot(fig)
    plt.close(fig)


def plot_distribution(df: pd.DataFrame, column: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df[column].dropna(), bins=40)
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("count")

    st.pyplot(fig)
    plt.close(fig)


def plot_confusion_matrix(pred_df: pd.DataFrame, title: str) -> None:
    cm = confusion_matrix(pred_df["is_anomaly"], pred_df["prediction"])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)

    st.pyplot(fig)
    plt.close(fig)


def plot_metric_bars(metrics_df: pd.DataFrame) -> None:
    for metric in ["precision", "recall", "f1_score", "pr_auc", "false_positives_per_1000"]:
        fig, ax = plt.subplots(figsize=(6, 4))
        subset = metrics_df[["model", metric]].copy()
        ax.bar(subset["model"], subset[metric])
        ax.set_title(metric)
        ax.set_ylabel(metric)

        st.pyplot(fig)
        plt.close(fig)


@st.cache_resource
def get_lstm_bundle():
    return load_lstm_bundle(MODEL_DIR)


def run_lstm_inference(input_df: pd.DataFrame) -> pd.DataFrame:
    bundle = get_lstm_bundle()
    return predict_with_lstm_bundle(input_df, bundle)
