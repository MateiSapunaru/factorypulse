from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from factorypulse.database.models import Base
from factorypulse.database.repository import (
    read_sensor_data,
    write_dataset_metadata,
    write_sensor_data,
)


@pytest.fixture()
def engine() -> Engine:
    return create_engine("sqlite:///:memory:")


def _sample_sensor_df() -> pd.DataFrame:
    base_time = datetime(2024, 1, 1)
    return pd.DataFrame(
        {
            "timestamp": [base_time + timedelta(minutes=i) for i in range(3)],
            "machine_id": ["m1", "m1", "m2"],
            "temperature": [70.0, 71.5, 68.2],
            "vibration": [0.5, 0.6, 0.4],
            "pressure": [101.0, 101.5, 100.8],
            "rpm": [1500.0, 1520.0, 1480.0],
            "power_draw": [12.0, 12.5, 11.8],
            "throughput": [95.0, 96.0, 94.0],
            "is_anomaly": [0, 1, 0],
            "anomaly_type": ["none", "spike", "none"],
        }
    )


def test_write_and_read_sensor_data_roundtrip(engine: Engine) -> None:
    df = _sample_sensor_df()
    write_sensor_data(df, engine, if_exists="replace")

    result = read_sensor_data(engine)

    assert len(result) == len(df)
    assert set(result["machine_id"]) == {"m1", "m2"}


def test_read_sensor_data_filters_by_machine_id(engine: Engine) -> None:
    df = _sample_sensor_df()
    write_sensor_data(df, engine, if_exists="replace")

    result = read_sensor_data(engine, machine_id="m1")

    assert len(result) == 2
    assert set(result["machine_id"]) == {"m1"}


def test_read_sensor_data_respects_limit(engine: Engine) -> None:
    df = _sample_sensor_df()
    write_sensor_data(df, engine, if_exists="replace")

    result = read_sensor_data(engine, limit=1)

    assert len(result) == 1


def test_write_dataset_metadata_inserts_row(engine: Engine) -> None:
    Base.metadata.create_all(engine)

    write_dataset_metadata(
        engine,
        dataset_name="synthetic-v1",
        dataset_version="0.1.0",
        source_path="data/processed/dataset.parquet",
        row_count=190_000,
        anomaly_count=24_700,
        notes="test insert",
    )

    with engine.connect() as conn:
        row = conn.execute(text("SELECT * FROM dataset_metadata")).mappings().first()

    assert row is not None
    assert row["dataset_name"] == "synthetic-v1"
    assert row["row_count"] == 190_000
