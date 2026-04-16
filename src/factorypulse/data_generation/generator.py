from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from factorypulse.data_generation.anomalies import apply_anomalies_to_machine
from factorypulse.data_generation.schemas import (
    AnomalyConfig,
    ContextualConfig,
    DatasetConfig,
    DriftConfig,
    GeneratorConfig,
    SignalConfig,
    SpikeConfig,
    StuckSensorConfig,
    VarianceIncreaseConfig,
)
from factorypulse.data_generation.signals import (
    build_time_components,
    build_time_index,
    generate_normal_signal,
)


def load_generator_config(config_path: str | Path) -> GeneratorConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    dataset = DatasetConfig(**raw_config["dataset"])
    signals = {
        name: SignalConfig(**signal_cfg)
        for name, signal_cfg in raw_config["signals"].items()
    }

    anomaly_raw = raw_config["anomalies"]
    anomalies = AnomalyConfig(
        anomaly_probability_per_machine=anomaly_raw["anomaly_probability_per_machine"],
        min_anomaly_length=anomaly_raw["min_anomaly_length"],
        max_anomaly_length=anomaly_raw["max_anomaly_length"],
        spike=SpikeConfig(**anomaly_raw["spike"]),
        drift=DriftConfig(**anomaly_raw["drift"]),
        stuck_sensor=StuckSensorConfig(**anomaly_raw["stuck_sensor"]),
        variance_increase=VarianceIncreaseConfig(**anomaly_raw["variance_increase"]),
        contextual=ContextualConfig(**anomaly_raw["contextual"]),
        target_features=anomaly_raw["target_features"],
    )

    return GeneratorConfig(
        random_seed=raw_config["random_seed"],
        dataset=dataset,
        signals=signals,
        anomalies=anomalies,
    )


def generate_base_dataset(config: GeneratorConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.random_seed)

    time_index = build_time_index(
        start_time=config.dataset.start_time,
        periods=config.dataset.periods,
        freq=config.dataset.freq,
    )
    time_components = build_time_components(time_index)

    frames: list[pd.DataFrame] = []

    for machine_idx in range(config.dataset.n_machines):
        machine_id = f"machine_{machine_idx:03d}"

        machine_df = pd.DataFrame(
            {
                "timestamp": time_index,
                "machine_id": machine_id,
            }
        )

        for feature_name, signal_config in config.signals.items():
            machine_df[feature_name] = generate_normal_signal(
                config=signal_config,
                time_components=time_components,
                rng=rng,
            )

        machine_df["is_anomaly"] = 0
        machine_df["anomaly_type"] = "none"

        machine_df = apply_anomalies_to_machine(
            machine_df=machine_df,
            signal_configs=config.signals,
            anomaly_config=config.anomalies,
            rng=rng,
        )

        frames.append(machine_df)

    df = pd.concat(frames, ignore_index=True)
    return df