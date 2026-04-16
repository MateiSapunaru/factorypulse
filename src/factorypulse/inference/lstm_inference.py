from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import load
from torch.utils.data import DataLoader

from factorypulse.models.deep.dataset import SequenceDataset, build_lstm_sequences
from factorypulse.models.deep.lstm_autoencoder import LSTMAutoencoder, LSTMAutoencoderConfig


@dataclass(frozen=True)
class LSTMBundle:
    model: LSTMAutoencoder
    scaler: object
    metadata: dict
    device: torch.device


def load_lstm_bundle(model_dir: str | Path) -> LSTMBundle:
    model_dir = Path(model_dir)

    metadata_path = model_dir / "model_metadata.json"
    scaler_path = model_dir / "scaler.joblib"
    weights_path = model_dir / "lstm_autoencoder.pt"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler file: {scaler_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing model weights file: {weights_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    scaler = load(scaler_path)

    model_config = LSTMAutoencoderConfig(
        input_size=int(metadata["input_size"]),
        hidden_size=int(metadata["hidden_size"]),
        latent_size=int(metadata["latent_size"]),
        num_layers=int(metadata["num_layers"]),
        dropout=float(metadata["dropout"]),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAutoencoder(model_config).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return LSTMBundle(
        model=model,
        scaler=scaler,
        metadata=metadata,
        device=device,
    )


def prepare_feature_dataframe(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    result = df.copy()

    missing_columns = [col for col in feature_columns if col not in result.columns]
    if missing_columns:
        raise ValueError(
            "Input dataframe is missing required feature columns: "
            + ", ".join(missing_columns)
        )

    for col in feature_columns:
        result[col] = result[col].astype(np.float32)

    result.loc[:, feature_columns] = bundle_safe_transform(
        result[feature_columns],
    )

    return result


def bundle_safe_transform(x: pd.DataFrame) -> pd.DataFrame:
    # Placeholder transformed later after scaler is available.
    # This function exists only to keep type/structure explicit.
    return x


def apply_scaler(
    df: pd.DataFrame,
    scaler: object,
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


def predict_with_lstm_bundle(
    df: pd.DataFrame,
    bundle: LSTMBundle,
    timestamp_column: str = "timestamp",
    machine_id_column: str = "machine_id",
    target_column: str = "is_anomaly",
) -> pd.DataFrame:
    metadata = bundle.metadata
    feature_columns = metadata["feature_columns"]
    sequence_length = int(metadata["sequence_length"])
    threshold = float(metadata["threshold"])

    result = df.copy()

    if timestamp_column not in result.columns:
        raise ValueError(f"Missing required column: {timestamp_column}")
    if machine_id_column not in result.columns:
        raise ValueError(f"Missing required column: {machine_id_column}")

    if target_column not in result.columns:
        result[target_column] = 0

    result[timestamp_column] = pd.to_datetime(result[timestamp_column])
    result = result.sort_values([machine_id_column, timestamp_column]).reset_index(drop=True)
    result = apply_scaler(result, bundle.scaler, feature_columns)

    sequences, seq_meta = build_lstm_sequences(
        df=result,
        feature_columns=feature_columns,
        machine_id_column=machine_id_column,
        timestamp_column=timestamp_column,
        target_column=target_column,
        sequence_length=sequence_length,
    )

    if len(sequences) == 0:
        return pd.DataFrame(
            columns=[
                machine_id_column,
                timestamp_column,
                target_column,
                "anomaly_score",
                "prediction",
            ]
        )

    dataset = SequenceDataset(sequences)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    scores = compute_reconstruction_errors(
        model=bundle.model,
        loader=loader,
        device=bundle.device,
    )

    pred_df = pd.DataFrame(
        {
            machine_id_column: seq_meta.machine_ids,
            timestamp_column: seq_meta.end_timestamps,
            target_column: seq_meta.labels,
            "anomaly_score": scores,
        }
    )
    pred_df["prediction"] = (pred_df["anomaly_score"] > threshold).astype(int)

    return pred_df