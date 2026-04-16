from __future__ import annotations

import numpy as np
import pandas as pd

from factorypulse.data_generation.schemas import SignalConfig


def build_time_index(start_time: str, periods: int, freq: str) -> pd.DatetimeIndex:
    """
    Create a regular timestamp index for the synthetic dataset.
    """
    return pd.date_range(start=start_time, periods=periods, freq=freq)


def build_time_components(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Extract normalized cyclical time components used to simulate seasonality.
    """
    hours = index.hour + index.minute / 60.0
    day_of_week = index.dayofweek

    return pd.DataFrame(
        {
            "hour_of_day": hours,
            "day_of_week": day_of_week,
            "time_step": np.arange(len(index), dtype=float),
        },
        index=index,
    )


def daily_seasonality(
    hour_of_day: np.ndarray,
    amplitude: float,
    phase_shift: float = 0.0,
) -> np.ndarray:
    """
    Simulate 24-hour periodic behavior.
    """
    return amplitude * np.sin((2.0 * np.pi * hour_of_day / 24.0) + phase_shift)


def weekly_seasonality(
    day_of_week: np.ndarray,
    amplitude: float,
    phase_shift: float = 0.0,
) -> np.ndarray:
    """
    Simulate 7-day periodic behavior.
    """
    return amplitude * np.sin((2.0 * np.pi * day_of_week / 7.0) + phase_shift)


def linear_trend(time_step: np.ndarray, slope: float) -> np.ndarray:
    """
    Simulate a slow long-term drift in normal operating behavior.
    """
    return slope * time_step


def gaussian_noise(size: int, std: float, rng: np.random.Generator) -> np.ndarray:
    """
    Add random local variability.
    """
    return rng.normal(loc=0.0, scale=std, size=size)


def generate_normal_signal(
    config: SignalConfig,
    time_components: pd.DataFrame,
    rng: np.random.Generator,
    machine_offset_scale: float = 1.0,
) -> np.ndarray:
    """
    Generate a single normal time-series signal for one machine.

    The signal is composed of:
    - machine-specific base offset
    - linear trend
    - daily seasonality
    - weekly seasonality
    - gaussian noise
    """
    n = len(time_components)

    machine_base = rng.normal(config.base_mean, config.base_std * machine_offset_scale)
    slope = rng.normal(0.0, config.trend_std)
    daily_phase = rng.uniform(0.0, 2.0 * np.pi)
    weekly_phase = rng.uniform(0.0, 2.0 * np.pi)

    trend = linear_trend(time_components["time_step"].to_numpy(), slope=slope)
    daily = daily_seasonality(
        time_components["hour_of_day"].to_numpy(),
        amplitude=config.daily_amplitude,
        phase_shift=daily_phase,
    )
    weekly = weekly_seasonality(
        time_components["day_of_week"].to_numpy(),
        amplitude=config.weekly_amplitude,
        phase_shift=weekly_phase,
    )
    noise = gaussian_noise(size=n, std=config.noise_std, rng=rng)

    signal = machine_base + trend + daily + weekly + noise
    return signal.astype(float)