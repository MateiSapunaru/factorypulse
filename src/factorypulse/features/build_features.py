from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from factorypulse.database.connection import get_engine
from factorypulse.database.repository import read_sensor_data


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float
    val_ratio: float
    test_ratio: float


@dataclass(frozen=True)
class FeatureConfig:
    timestamp_column: str
    machine_id_column: str
    target_column: str
    base_feature_columns: list[str]
    lag_steps: list[int]
    rolling_windows: list[int]


@dataclass(frozen=True)
class TrainingConfig:
    split: SplitConfig
    features: FeatureConfig


def load_training_config(config_path: str | Path) -> TrainingConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    split = SplitConfig(**raw_config["split"])
    features = FeatureConfig(**raw_config["features"])

    total = split.train_ratio + split.val_ratio + split.test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, but got {total:.4f}")

    return TrainingConfig(split=split, features=features)


def load_dataset_from_postgres() -> pd.DataFrame:
    engine = get_engine()
    return read_sensor_data(engine=engine)


def validate_time_series_dataset(
    df: pd.DataFrame,
    timestamp_column: str,
    machine_id_column: str,
) -> None:
    if df.empty:
        raise ValueError("Input dataframe is empty.")

    required_columns = {timestamp_column, machine_id_column}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if df[timestamp_column].isna().any():
        raise ValueError(f"Column '{timestamp_column}' contains null values.")

    if df[machine_id_column].isna().any():
        raise ValueError(f"Column '{machine_id_column}' contains null values.")

    duplicates = df.duplicated(subset=[machine_id_column, timestamp_column]).sum()
    if duplicates > 0:
        raise ValueError(f"Found {duplicates} duplicate (machine_id, timestamp) rows.")


def sort_time_series_dataset(
    df: pd.DataFrame,
    timestamp_column: str,
    machine_id_column: str,
) -> pd.DataFrame:
    return (
        df.copy()
        .sort_values([machine_id_column, timestamp_column])
        .reset_index(drop=True)
    )


def temporal_split(
    df: pd.DataFrame,
    timestamp_column: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unique_timestamps = (
        pd.Series(df[timestamp_column].sort_values().unique())
        .sort_values()
        .reset_index(drop=True)
    )

    n_timestamps = len(unique_timestamps)
    train_end = int(n_timestamps * train_ratio)
    val_end = train_end + int(n_timestamps * val_ratio)

    train_timestamps = unique_timestamps.iloc[:train_end]
    val_timestamps = unique_timestamps.iloc[train_end:val_end]
    test_timestamps = unique_timestamps.iloc[val_end:]

    train_df = df[df[timestamp_column].isin(train_timestamps)].copy()
    val_df = df[df[timestamp_column].isin(val_timestamps)].copy()
    test_df = df[df[timestamp_column].isin(test_timestamps)].copy()

    return train_df, val_df, test_df


def build_temporal_features(
    df: pd.DataFrame,
    timestamp_column: str,
) -> pd.DataFrame:
    result = df.copy()
    ts = pd.to_datetime(result[timestamp_column])

    result["hour"] = ts.dt.hour
    result["day_of_week"] = ts.dt.dayofweek
    result["is_weekend"] = (result["day_of_week"] >= 5).astype(int)

    return result


def build_lag_features(
    df: pd.DataFrame,
    machine_id_column: str,
    feature_columns: list[str],
    lag_steps: list[int],
) -> pd.DataFrame:
    result = df.copy()

    for feature in feature_columns:
        for lag in lag_steps:
            result[f"{feature}_lag_{lag}"] = (
                result.groupby(machine_id_column)[feature].shift(lag)
            )

    return result


def build_rolling_features(
    df: pd.DataFrame,
    machine_id_column: str,
    feature_columns: list[str],
    rolling_windows: list[int],
) -> pd.DataFrame:
    result = df.copy()

    for feature in feature_columns:
        grouped = result.groupby(machine_id_column)[feature]

        for window in rolling_windows:
            result[f"{feature}_roll_mean_{window}"] = (
                grouped.transform(lambda s: s.rolling(window=window, min_periods=window).mean())
            )
            result[f"{feature}_roll_std_{window}"] = (
                grouped.transform(lambda s: s.rolling(window=window, min_periods=window).std())
            )
            result[f"{feature}_roll_min_{window}"] = (
                grouped.transform(lambda s: s.rolling(window=window, min_periods=window).min())
            )
            result[f"{feature}_roll_max_{window}"] = (
                grouped.transform(lambda s: s.rolling(window=window, min_periods=window).max())
            )

    return result


def drop_rows_with_feature_nans(
    df: pd.DataFrame,
    protected_columns: list[str],
) -> pd.DataFrame:
    feature_columns = [col for col in df.columns if col not in protected_columns]
    return df.dropna(subset=feature_columns).reset_index(drop=True)


def summarize_split(
    name: str,
    df: pd.DataFrame,
    timestamp_column: str,
    target_column: str,
) -> None:
    anomaly_count = int(df[target_column].sum())
    anomaly_ratio = anomaly_count / len(df) if len(df) > 0 else 0.0

    print(f"\n{name.upper()} SPLIT")
    print(f"Rows: {len(df)}")
    print(f"Start: {df[timestamp_column].min()}")
    print(f"End:   {df[timestamp_column].max()}")
    print(f"Anomalies: {anomaly_count}")
    print(f"Anomaly ratio: {anomaly_ratio:.4f}")
    print(f"Columns: {len(df.columns)}")