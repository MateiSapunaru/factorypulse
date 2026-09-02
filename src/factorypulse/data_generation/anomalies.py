from __future__ import annotations

import numpy as np
import pandas as pd

from factorypulse.data_generation.schemas import AnomalyConfig, SignalConfig


def inject_spike(
    values: np.ndarray,
    start: int,
    end: int,
    signal_config: SignalConfig,
    rng: np.random.Generator,
    magnitude_std_factor: float,
) -> np.ndarray:
    result = values.copy()
    spike_idx = rng.integers(start, end)
    direction = rng.choice([-1.0, 1.0])
    magnitude = direction * magnitude_std_factor * signal_config.noise_std
    result[spike_idx] += magnitude
    return result


def inject_drift(
    values: np.ndarray,
    start: int,
    end: int,
    signal_config: SignalConfig,
    rng: np.random.Generator,
    slope_std_factor: float,
) -> np.ndarray:
    result = values.copy()
    length = end - start
    direction = rng.choice([-1.0, 1.0])
    slope = direction * slope_std_factor * signal_config.base_std
    drift = np.linspace(0.0, slope, num=length)
    result[start:end] += drift
    return result


def inject_stuck_sensor(
    values: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    result = values.copy()
    stuck_value = result[start]
    result[start:end] = stuck_value
    return result


def inject_variance_increase(
    values: np.ndarray,
    start: int,
    end: int,
    signal_config: SignalConfig,
    rng: np.random.Generator,
    noise_multiplier: float,
) -> np.ndarray:
    result = values.copy()
    extra_noise = rng.normal(
        loc=0.0,
        scale=signal_config.noise_std * noise_multiplier,
        size=end - start,
    )
    result[start:end] += extra_noise
    return result


def inject_contextual(
    values: np.ndarray,
    timestamps: pd.Series,
    start: int,
    end: int,
    signal_config: SignalConfig,
    magnitude_std_factor: float,
) -> np.ndarray:
    result = values.copy()

    hours = timestamps.iloc[start:end].dt.hour.to_numpy()
    night_mask = (hours <= 5) | (hours >= 22)

    if night_mask.any():
        result[start:end][night_mask] += magnitude_std_factor * signal_config.noise_std
    else:
        result[start:end] -= magnitude_std_factor * signal_config.noise_std

    return result


def sample_anomaly_segments(
    n_points: int,
    anomaly_probability: float,
    min_length: int,
    max_length: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    starts = np.where(rng.random(n_points) < anomaly_probability)[0]
    segments: list[tuple[int, int]] = []

    last_end = -1
    for start in starts:
        if start < last_end:
            continue

        length = int(rng.integers(min_length, max_length + 1))
        end = min(start + length, n_points)

        segments.append((start, end))
        last_end = end

    return segments


def apply_anomalies_to_machine(
    machine_df: pd.DataFrame,
    signal_configs: dict[str, SignalConfig],
    anomaly_config: AnomalyConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    result = machine_df.copy()
    n_points = len(result)

    segments = sample_anomaly_segments(
        n_points=n_points,
        anomaly_probability=anomaly_config.anomaly_probability_per_machine,
        min_length=anomaly_config.min_anomaly_length,
        max_length=anomaly_config.max_anomaly_length,
        rng=rng,
    )

    enabled_anomaly_types: list[str] = []
    if anomaly_config.spike.enabled:
        enabled_anomaly_types.append("spike")
    if anomaly_config.drift.enabled:
        enabled_anomaly_types.append("drift")
    if anomaly_config.stuck_sensor.enabled:
        enabled_anomaly_types.append("stuck_sensor")
    if anomaly_config.variance_increase.enabled:
        enabled_anomaly_types.append("variance_increase")
    if anomaly_config.contextual.enabled:
        enabled_anomaly_types.append("contextual")

    for start, end in segments:
        anomaly_type = rng.choice(enabled_anomaly_types)
        feature = rng.choice(anomaly_config.target_features)

        signal_config = signal_configs[feature]
        values = result[feature].to_numpy()

        if anomaly_type == "spike":
            updated = inject_spike(
                values=values,
                start=start,
                end=end,
                signal_config=signal_config,
                rng=rng,
                magnitude_std_factor=anomaly_config.spike.magnitude_std_factor,
            )
        elif anomaly_type == "drift":
            updated = inject_drift(
                values=values,
                start=start,
                end=end,
                signal_config=signal_config,
                rng=rng,
                slope_std_factor=anomaly_config.drift.slope_std_factor,
            )
        elif anomaly_type == "stuck_sensor":
            updated = inject_stuck_sensor(
                values=values,
                start=start,
                end=end,
            )
        elif anomaly_type == "variance_increase":
            updated = inject_variance_increase(
                values=values,
                start=start,
                end=end,
                signal_config=signal_config,
                rng=rng,
                noise_multiplier=anomaly_config.variance_increase.noise_multiplier,
            )
        elif anomaly_type == "contextual":
            updated = inject_contextual(
                values=values,
                timestamps=result["timestamp"],
                start=start,
                end=end,
                signal_config=signal_config,
                magnitude_std_factor=anomaly_config.contextual.magnitude_std_factor,
            )
        else:
            continue

        result[feature] = updated
        result.loc[start : end - 1, "is_anomaly"] = 1
        result.loc[start : end - 1, "anomaly_type"] = anomaly_type

    return result
