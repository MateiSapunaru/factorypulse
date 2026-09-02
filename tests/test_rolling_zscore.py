from __future__ import annotations

import pandas as pd

from factorypulse.models.baseline.rolling_zscore import (
    RollingZScoreConfig,
    compute_rolling_zscore_predictions,
)


def test_stable_signal_has_no_anomalies() -> None:
    df = pd.DataFrame({"machine_id": ["m1"] * 20, "temperature": [50.0] * 20})
    config = RollingZScoreConfig(
        feature_columns=["temperature"],
        window_size=5,
        z_threshold=3.0,
    )

    result = compute_rolling_zscore_predictions(df, config)

    assert result["prediction"].eq(0).all()


def test_flags_large_deviation_as_anomaly() -> None:
    values = [50.0] * 15 + [100_000.0] + [50.0] * 4
    df = pd.DataFrame({"machine_id": ["m1"] * len(values), "temperature": values})
    config = RollingZScoreConfig(
        feature_columns=["temperature"],
        window_size=10,
        z_threshold=2.5,
    )

    result = compute_rolling_zscore_predictions(df, config)

    assert result.loc[15, "prediction"] == 1
    assert result.loc[:14, "prediction"].eq(0).all()


def test_rolling_window_does_not_leak_across_machines() -> None:
    df = pd.DataFrame(
        {
            "machine_id": ["m1"] * 10 + ["m2"] * 10,
            "temperature": [50.0] * 10 + [50.0] * 10,
        }
    )
    config = RollingZScoreConfig(
        feature_columns=["temperature"],
        window_size=5,
        z_threshold=3.0,
    )

    result = compute_rolling_zscore_predictions(df, config)

    assert result["prediction"].eq(0).all()


def test_multiple_feature_columns_combine_into_anomaly_score() -> None:
    df = pd.DataFrame(
        {
            "machine_id": ["m1"] * 20,
            "temperature": [50.0] * 20,
            "vibration": [1.0] * 15 + [100_000.0] + [1.0] * 4,
        }
    )
    config = RollingZScoreConfig(
        feature_columns=["temperature", "vibration"],
        window_size=10,
        z_threshold=2.5,
    )

    result = compute_rolling_zscore_predictions(df, config)

    assert result.loc[15, "prediction"] == 1
    assert result.loc[15, "anomaly_score"] == result.loc[15, "vibration_zscore"]
