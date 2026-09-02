from __future__ import annotations

from pathlib import Path

from factorypulse.features.build_features import (
    build_lag_features,
    build_rolling_features,
    build_temporal_features,
    drop_rows_with_feature_nans,
    load_dataset_from_postgres,
    load_training_config,
    sort_time_series_dataset,
    summarize_split,
    temporal_split,
    validate_time_series_dataset,
)


def main() -> None:
    config = load_training_config(Path("configs/training.yaml"))

    df = load_dataset_from_postgres()
    df["timestamp"] = df["timestamp"].astype("datetime64[ns]")

    validate_time_series_dataset(
        df=df,
        timestamp_column=config.features.timestamp_column,
        machine_id_column=config.features.machine_id_column,
    )

    df = sort_time_series_dataset(
        df=df,
        timestamp_column=config.features.timestamp_column,
        machine_id_column=config.features.machine_id_column,
    )

    df = build_temporal_features(
        df=df,
        timestamp_column=config.features.timestamp_column,
    )

    df = build_lag_features(
        df=df,
        machine_id_column=config.features.machine_id_column,
        feature_columns=config.features.base_feature_columns,
        lag_steps=config.features.lag_steps,
    )

    df = build_rolling_features(
        df=df,
        machine_id_column=config.features.machine_id_column,
        feature_columns=config.features.base_feature_columns,
        rolling_windows=config.features.rolling_windows,
    )

    protected_columns = [
        config.features.timestamp_column,
        config.features.machine_id_column,
        config.features.target_column,
        "anomaly_type",
    ]

    df = drop_rows_with_feature_nans(
        df=df,
        protected_columns=protected_columns,
    )

    train_df, val_df, test_df = temporal_split(
        df=df,
        timestamp_column=config.features.timestamp_column,
        train_ratio=config.split.train_ratio,
        val_ratio=config.split.val_ratio,
        test_ratio=config.split.test_ratio,
    )

    summarize_split(
        name="train",
        df=train_df,
        timestamp_column=config.features.timestamp_column,
        target_column=config.features.target_column,
    )
    summarize_split(
        name="validation",
        df=val_df,
        timestamp_column=config.features.timestamp_column,
        target_column=config.features.target_column,
    )
    summarize_split(
        name="test",
        df=test_df,
        timestamp_column=config.features.timestamp_column,
        target_column=config.features.target_column,
    )

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    train_df.to_csv("data/processed/train_features.csv", index=False)
    val_df.to_csv("data/processed/val_features.csv", index=False)
    test_df.to_csv("data/processed/test_features.csv", index=False)

    print("\nSaved feature datasets to data/processed/")


if __name__ == "__main__":
    main()
