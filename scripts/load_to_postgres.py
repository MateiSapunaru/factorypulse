from __future__ import annotations

import pandas as pd

from factorypulse.database.connection import get_engine
from factorypulse.database.repository import write_dataset_metadata, write_sensor_data


def main() -> None:
    input_path = "data/raw/synthetic_data.csv"
    dataset_name = "synthetic_industrial_timeseries"
    dataset_version = "v1"

    df = pd.read_csv(input_path, parse_dates=["timestamp"])
    engine = get_engine()

    write_sensor_data(df=df, engine=engine, if_exists="replace", chunksize=10000)

    anomaly_count = int(df["is_anomaly"].sum())
    write_dataset_metadata(
        engine=engine,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        source_path=input_path,
        row_count=len(df),
        anomaly_count=anomaly_count,
        notes="Synthetic multivariate industrial dataset with injected anomalies.",
    )

    print(f"Loaded {len(df)} rows into PostgreSQL.")
    print(f"Logged dataset metadata for {dataset_name} ({dataset_version}).")


if __name__ == "__main__":
    main()