from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RollingZScoreConfig:
    feature_columns: list[str]
    window_size: int
    z_threshold: float
    machine_id_column: str = "machine_id"


def compute_rolling_zscore_predictions(
    df: pd.DataFrame,
    config: RollingZScoreConfig,
) -> pd.DataFrame:
    result = df.copy()
    prediction_columns: list[str] = []
    zscore_columns: list[str] = []

    for feature in config.feature_columns:
        group = result.groupby(config.machine_id_column)[feature]

        rolling_mean = group.transform(
            lambda s: s.rolling(window=config.window_size, min_periods=config.window_size).mean()
        )
        rolling_std = group.transform(
            lambda s: s.rolling(window=config.window_size, min_periods=config.window_size).std()
        )

        z_score = (result[feature] - rolling_mean) / rolling_std.replace(0, np.nan)
        z_score = z_score.abs().fillna(0.0)

        z_col = f"{feature}_zscore"
        pred_col = f"{feature}_zscore_pred"

        result[z_col] = z_score
        result[pred_col] = (z_score > config.z_threshold).astype(int)

        zscore_columns.append(z_col)
        prediction_columns.append(pred_col)

    result["anomaly_score"] = result[zscore_columns].max(axis=1)
    result["prediction"] = (result["anomaly_score"] > config.z_threshold).astype(int)

    return result
