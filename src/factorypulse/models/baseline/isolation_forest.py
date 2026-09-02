from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class IsolationForestConfig:
    n_estimators: int
    contamination: float
    random_state: int
    use_scaling: bool = True


def train_isolation_forest(
    df: pd.DataFrame,
    feature_columns: list[str],
    config: IsolationForestConfig,
) -> tuple[IsolationForest, StandardScaler | None]:
    x_train = df[feature_columns].copy()

    scaler = None
    if config.use_scaling:
        scaler = StandardScaler()
        x_train = pd.DataFrame(
            scaler.fit_transform(x_train),
            columns=feature_columns,
            index=df.index,
        )

    model = IsolationForest(
        n_estimators=config.n_estimators,
        contamination=config.contamination,
        random_state=config.random_state,
        n_jobs=-1,
    )
    model.fit(x_train)

    return model, scaler


def predict_isolation_forest(
    df: pd.DataFrame,
    model: IsolationForest,
    feature_columns: list[str],
    scaler: StandardScaler | None = None,
) -> pd.DataFrame:
    result = df.copy()
    x = df[feature_columns].copy()

    if scaler is not None:
        x = pd.DataFrame(
            scaler.transform(x),
            columns=feature_columns,
            index=df.index,
        )

    scores = model.decision_function(x)
    preds = model.predict(x)

    result["anomaly_score"] = -scores
    result["prediction"] = (preds == -1).astype(int)

    return result
