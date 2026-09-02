from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SignalConfig:
    base_mean: float
    base_std: float
    trend_std: float
    daily_amplitude: float
    weekly_amplitude: float
    noise_std: float


@dataclass(frozen=True)
class DatasetConfig:
    n_machines: int
    start_time: str
    periods: int
    freq: str


@dataclass(frozen=True)
class SpikeConfig:
    enabled: bool
    magnitude_std_factor: float


@dataclass(frozen=True)
class DriftConfig:
    enabled: bool
    slope_std_factor: float


@dataclass(frozen=True)
class StuckSensorConfig:
    enabled: bool


@dataclass(frozen=True)
class VarianceIncreaseConfig:
    enabled: bool
    noise_multiplier: float


@dataclass(frozen=True)
class ContextualConfig:
    enabled: bool
    magnitude_std_factor: float


@dataclass(frozen=True)
class AnomalyConfig:
    anomaly_probability_per_machine: float
    min_anomaly_length: int
    max_anomaly_length: int
    spike: SpikeConfig
    drift: DriftConfig
    stuck_sensor: StuckSensorConfig
    variance_increase: VarianceIncreaseConfig
    contextual: ContextualConfig
    target_features: list[str]


@dataclass(frozen=True)
class GeneratorConfig:
    random_seed: int
    dataset: DatasetConfig
    signals: dict[str, SignalConfig]
    anomalies: AnomalyConfig
