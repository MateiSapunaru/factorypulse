from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from joblib import dump
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from factorypulse.models.deep.dataset import SequenceDataset, build_lstm_sequences
from factorypulse.models.deep.lstm_autoencoder import LSTMAutoencoder, LSTMAutoencoderConfig
from factorypulse.training.evaluate import evaluate_binary_predictions, print_evaluation_result
from factorypulse.training.mlflow_utils import (
    log_artifact_if_exists,
    log_metrics_with_prefix,
    log_params_flat,
    log_text_artifact,
    save_confusion_matrix_plot,
    save_pr_curve_plot,
    save_timeseries_anomaly_plot,
    setup_mlflow,
)


@dataclass(frozen=True)
class MLflowSection:
    tracking_uri: str
    experiment_name: str
    run_name_prefix: str


@dataclass(frozen=True)
class LSTMSection:
    sequence_length: int
    hidden_size: int
    latent_size: int
    num_layers: int
    dropout: float
    batch_size: int
    learning_rate: float
    num_epochs: int
    threshold_percentile: float
    random_state: int
    score_mode: str = "mse"
    threshold_search_points: int = 120
    max_fp_per_1000: float = 150.0


@dataclass(frozen=True)
class TrainingFileConfig:
    mlflow: MLflowSection
    lstm_autoencoder: LSTMSection


def load_config(config_path: str | Path) -> TrainingFileConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return TrainingFileConfig(
        mlflow=MLflowSection(**raw["mlflow"]),
        lstm_autoencoder=LSTMSection(**raw["lstm_autoencoder"]),
    )


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if col not in ["timestamp", "machine_id", "is_anomaly", "anomaly_type"]
    ]


def fit_scaler_on_normal_train(
    train_df: pd.DataFrame,
    feature_columns: list[str],
) -> StandardScaler:
    normal_train = train_df[train_df["is_anomaly"] == 0].copy()
    x = normal_train[feature_columns].astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(x)
    return scaler


def apply_scaler(
    df: pd.DataFrame,
    scaler: StandardScaler,
    feature_columns: list[str],
) -> pd.DataFrame:
    result = df.copy()

    for col in feature_columns:
        result[col] = result[col].astype(np.float32)

    scaled = scaler.transform(result[feature_columns].astype(np.float32))
    scaled_df = pd.DataFrame(scaled, columns=feature_columns, index=result.index)

    result.loc[:, feature_columns] = scaled_df
    return result


def compute_reconstruction_errors(
    model: LSTMAutoencoder,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    errors: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            reconstructed = model(batch)
            batch_error = torch.mean((reconstructed - batch) ** 2, dim=(1, 2))
            errors.append(batch_error.cpu().numpy())

    return np.concatenate(errors) if errors else np.array([], dtype=np.float32)


def train_model(
    model: LSTMAutoencoder,
    loader: DataLoader,
    device: torch.device,
    learning_rate: float,
    num_epochs: int,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    history: list[float] = []

    for epoch in range(num_epochs):
        model.train()
        epoch_losses: list[float] = []

        for batch in loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.item()))

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        history.append(mean_loss)
        print(f"Epoch {epoch + 1}/{num_epochs} - train_loss: {mean_loss:.6f}")

    return history


def build_prediction_dataframe(
    metadata_machine_ids: list[str],
    metadata_timestamps: list[pd.Timestamp],
    metadata_labels: list[int],
    scores: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    pred_df = pd.DataFrame(
        {
            "machine_id": metadata_machine_ids,
            "timestamp": metadata_timestamps,
            "is_anomaly": metadata_labels,
            "anomaly_score": scores,
        }
    )
    pred_df["prediction"] = (pred_df["anomaly_score"] > threshold).astype(int)
    return pred_df


def save_loss_curve(
    loss_history: list[float],
    output_path: str | Path,
) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(loss_history) + 1), loss_history, linewidth=1.8)
    ax.set_title("LSTM Autoencoder Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def save_threshold_search_plot(
    threshold_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(threshold_df["threshold"], threshold_df["f1_score"], label="F1 score")
    ax1.plot(threshold_df["threshold"], threshold_df["precision"], label="Precision")
    ax1.plot(threshold_df["threshold"], threshold_df["recall"], label="Recall")
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Metric value")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(
        threshold_df["threshold"],
        threshold_df["false_positives_per_1000"],
        linestyle="--",
        label="FP/1000",
    )
    ax2.set_ylabel("False positives per 1000")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def search_best_threshold_with_fp_constraint(
    y_true: np.ndarray,
    scores: np.ndarray,
    num_thresholds: int,
    max_fp_per_1000: float,
) -> tuple[float, float, pd.DataFrame]:
    if len(scores) == 0:
        raise ValueError("Scores array is empty.")

    score_min = float(scores.min())
    score_max = float(scores.max())

    if np.isclose(score_min, score_max):
        candidate_thresholds = np.array([score_min], dtype=float)
    else:
        candidate_thresholds = np.linspace(score_min, score_max, num_thresholds)

    rows: list[dict[str, float]] = []

    for threshold in candidate_thresholds:
        preds = (scores > threshold).astype(int)
        result = evaluate_binary_predictions(
            y_true=pd.Series(y_true),
            y_pred=pd.Series(preds),
            y_score=pd.Series(scores),
        )
        rows.append(
            {
                "threshold": float(threshold),
                "precision": result.precision,
                "recall": result.recall,
                "f1_score": result.f1_score,
                "pr_auc": result.pr_auc,
                "false_positives_per_1000": result.false_positives_per_1000,
            }
        )

    threshold_df = pd.DataFrame(rows)

    valid_df = threshold_df[threshold_df["false_positives_per_1000"] <= max_fp_per_1000].copy()

    if not valid_df.empty:
        best_row = valid_df.sort_values(
            by=["f1_score", "precision", "recall"],
            ascending=[False, False, False],
        ).iloc[0]
    else:
        best_row = threshold_df.sort_values(
            by=["false_positives_per_1000", "f1_score"],
            ascending=[True, False],
        ).iloc[0]

    return float(best_row["threshold"]), float(best_row["f1_score"]), threshold_df


def main() -> None:
    config = load_config("configs/training.yaml")
    torch.manual_seed(config.lstm_autoencoder.random_state)
    np.random.seed(config.lstm_autoencoder.random_state)

    setup_mlflow(
        tracking_uri=config.mlflow.tracking_uri,
        experiment_name="factorypulse_lstm_autoencoder",
    )

    train_df = pd.read_csv("data/processed/train_features.csv", parse_dates=["timestamp"])
    val_df = pd.read_csv("data/processed/val_features.csv", parse_dates=["timestamp"])
    test_df = pd.read_csv("data/processed/test_features.csv", parse_dates=["timestamp"])

    feature_columns = select_feature_columns(train_df)

    scaler = fit_scaler_on_normal_train(train_df, feature_columns)
    train_scaled = apply_scaler(train_df, scaler, feature_columns)
    val_scaled = apply_scaler(val_df, scaler, feature_columns)
    test_scaled = apply_scaler(test_df, scaler, feature_columns)

    sequence_length = config.lstm_autoencoder.sequence_length

    train_sequences_all, train_meta_all = build_lstm_sequences(
        df=train_scaled,
        feature_columns=feature_columns,
        machine_id_column="machine_id",
        timestamp_column="timestamp",
        target_column="is_anomaly",
        sequence_length=sequence_length,
    )
    val_sequences, val_meta = build_lstm_sequences(
        df=val_scaled,
        feature_columns=feature_columns,
        machine_id_column="machine_id",
        timestamp_column="timestamp",
        target_column="is_anomaly",
        sequence_length=sequence_length,
    )
    test_sequences, test_meta = build_lstm_sequences(
        df=test_scaled,
        feature_columns=feature_columns,
        machine_id_column="machine_id",
        timestamp_column="timestamp",
        target_column="is_anomaly",
        sequence_length=sequence_length,
    )

    train_labels_all = np.array(train_meta_all.labels, dtype=int)
    train_normal_mask = train_labels_all == 0
    train_sequences = train_sequences_all[train_normal_mask]

    train_dataset = SequenceDataset(train_sequences)
    train_score_dataset = SequenceDataset(train_sequences_all)
    val_dataset = SequenceDataset(val_sequences)
    test_dataset = SequenceDataset(test_sequences)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.lstm_autoencoder.batch_size,
        shuffle=True,
    )
    train_score_loader = DataLoader(
        train_score_dataset,
        batch_size=config.lstm_autoencoder.batch_size,
        shuffle=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.lstm_autoencoder.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.lstm_autoencoder.batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = LSTMAutoencoderConfig(
        input_size=len(feature_columns),
        hidden_size=config.lstm_autoencoder.hidden_size,
        latent_size=config.lstm_autoencoder.latent_size,
        num_layers=config.lstm_autoencoder.num_layers,
        dropout=config.lstm_autoencoder.dropout,
    )
    model = LSTMAutoencoder(model_config).to(device)

    artifact_root = Path("data/artifacts")
    prediction_dir = artifact_root / "predictions"
    plot_dir = artifact_root / "plots"
    model_dir = artifact_root / "models"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(
        run_name=f"{config.mlflow.run_name_prefix}_lstm_autoencoder"
    ):
        log_params_flat(
            {
                "model_name": "lstm_autoencoder",
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "test_rows": len(test_df),
                "train_sequence_count_normal_only": len(train_sequences),
                "train_sequence_count_all": len(train_sequences_all),
                "val_sequence_count": len(val_sequences),
                "test_sequence_count": len(test_sequences),
                "feature_count": len(feature_columns),
            }
        )
        log_params_flat(config.lstm_autoencoder.__dict__, prefix="lstm_autoencoder")

        loss_history = train_model(
            model=model,
            loader=train_loader,
            device=device,
            learning_rate=config.lstm_autoencoder.learning_rate,
            num_epochs=config.lstm_autoencoder.num_epochs,
        )

        mlflow.log_metric("final_train_loss", loss_history[-1] if loss_history else 0.0)

        loss_curve_path = plot_dir / "lstm_autoencoder_loss_curve.png"
        save_loss_curve(loss_history, loss_curve_path)
        log_artifact_if_exists(loss_curve_path)

        train_scores_all = compute_reconstruction_errors(model, train_score_loader, device)
        val_scores = compute_reconstruction_errors(model, val_loader, device)
        test_scores = compute_reconstruction_errors(model, test_loader, device)

        threshold, best_val_f1_from_search, threshold_df = search_best_threshold_with_fp_constraint(
            y_true=np.array(val_meta.labels, dtype=int),
            scores=val_scores,
            num_thresholds=config.lstm_autoencoder.threshold_search_points,
            max_fp_per_1000=config.lstm_autoencoder.max_fp_per_1000,
        )

        percentile_threshold = float(
            np.percentile(
                train_scores_all,
                config.lstm_autoencoder.threshold_percentile,
            )
        )

        mlflow.log_param("lstm_autoencoder.threshold_value_constrained", threshold)
        mlflow.log_param("lstm_autoencoder.threshold_value_train_percentile", percentile_threshold)
        mlflow.log_metric("val_best_f1_from_threshold_search", best_val_f1_from_search)

        threshold_csv_path = prediction_dir / "lstm_autoencoder_threshold_search.csv"
        threshold_df.to_csv(threshold_csv_path, index=False)
        log_artifact_if_exists(threshold_csv_path)

        threshold_plot_path = plot_dir / "lstm_autoencoder_threshold_search.png"
        save_threshold_search_plot(threshold_df, threshold_plot_path)
        log_artifact_if_exists(threshold_plot_path)

        train_pred_df = build_prediction_dataframe(
            metadata_machine_ids=train_meta_all.machine_ids,
            metadata_timestamps=train_meta_all.end_timestamps,
            metadata_labels=train_meta_all.labels,
            scores=train_scores_all,
            threshold=threshold,
        )
        val_pred_df = build_prediction_dataframe(
            metadata_machine_ids=val_meta.machine_ids,
            metadata_timestamps=val_meta.end_timestamps,
            metadata_labels=val_meta.labels,
            scores=val_scores,
            threshold=threshold,
        )
        test_pred_df = build_prediction_dataframe(
            metadata_machine_ids=test_meta.machine_ids,
            metadata_timestamps=test_meta.end_timestamps,
            metadata_labels=test_meta.labels,
            scores=test_scores,
            threshold=threshold,
        )

        train_result = evaluate_binary_predictions(
            y_true=train_pred_df["is_anomaly"],
            y_pred=train_pred_df["prediction"],
            y_score=train_pred_df["anomaly_score"],
        )
        val_result = evaluate_binary_predictions(
            y_true=val_pred_df["is_anomaly"],
            y_pred=val_pred_df["prediction"],
            y_score=val_pred_df["anomaly_score"],
        )
        test_result = evaluate_binary_predictions(
            y_true=test_pred_df["is_anomaly"],
            y_pred=test_pred_df["prediction"],
            y_score=test_pred_df["anomaly_score"],
        )

        print_evaluation_result("LSTM Autoencoder - Train", train_result)
        print_evaluation_result("LSTM Autoencoder - Validation", val_result)
        print_evaluation_result("LSTM Autoencoder - Test", test_result)

        log_metrics_with_prefix(train_result.to_dict(), prefix="train")
        log_metrics_with_prefix(val_result.to_dict(), prefix="val")
        log_metrics_with_prefix(test_result.to_dict(), prefix="test")

        pred_path = prediction_dir / "lstm_autoencoder_test_predictions.csv"
        test_pred_df.to_csv(pred_path, index=False)
        log_artifact_if_exists(pred_path)

        cm_path = plot_dir / "lstm_autoencoder_confusion_matrix.png"
        save_confusion_matrix_plot(
            y_true=test_pred_df["is_anomaly"],
            y_pred=test_pred_df["prediction"],
            output_path=cm_path,
            title="LSTM Autoencoder - Test Confusion Matrix",
        )
        log_artifact_if_exists(cm_path)

        pr_path = plot_dir / "lstm_autoencoder_pr_curve.png"
        save_pr_curve_plot(
            y_true=test_pred_df["is_anomaly"],
            y_score=test_pred_df["anomaly_score"],
            output_path=pr_path,
            title="LSTM Autoencoder - Test PR Curve",
        )
        log_artifact_if_exists(pr_path)

        ts_plot_df = test_pred_df[test_pred_df["machine_id"] == "machine_000"].copy()
        if not ts_plot_df.empty:
            base_signal_lookup = test_df[["machine_id", "timestamp", "temperature"]].copy()
            ts_plot_df = ts_plot_df.merge(
                base_signal_lookup,
                on=["machine_id", "timestamp"],
                how="left",
            )

            ts_path = plot_dir / "lstm_autoencoder_timeseries_machine_000_temperature.png"
            save_timeseries_anomaly_plot(
                df=ts_plot_df,
                timestamp_column="timestamp",
                value_column="temperature",
                true_label_column="is_anomaly",
                pred_label_column="prediction",
                output_path=ts_path,
                title="LSTM Autoencoder - machine_000 - temperature",
            )
            log_artifact_if_exists(ts_path)

        # Save local deployable bundle for dashboard inference
        model_path = model_dir / "lstm_autoencoder.pt"
        torch.save(model.state_dict(), model_path)
        log_artifact_if_exists(model_path)

        scaler_path = model_dir / "scaler.joblib"
        dump(scaler, scaler_path)
        log_artifact_if_exists(scaler_path)

        metadata = {
            "model_type": "lstm_autoencoder",
            "input_size": len(feature_columns),
            "feature_columns": feature_columns,
            "sequence_length": config.lstm_autoencoder.sequence_length,
            "hidden_size": config.lstm_autoencoder.hidden_size,
            "latent_size": config.lstm_autoencoder.latent_size,
            "num_layers": config.lstm_autoencoder.num_layers,
            "dropout": config.lstm_autoencoder.dropout,
            "threshold": threshold,
            "threshold_train_percentile_reference": percentile_threshold,
            "score_mode": config.lstm_autoencoder.score_mode,
            "max_fp_per_1000": config.lstm_autoencoder.max_fp_per_1000,
        }

        metadata_path = model_dir / "model_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        log_artifact_if_exists(metadata_path)

        log_text_artifact(
            content=(
                "Local LSTM autoencoder bundle for dashboard inference. "
                "Includes model weights, scaler, and metadata."
            ),
            artifact_file="notes/lstm_autoencoder_summary.txt",
        )


if __name__ == "__main__":
    main()