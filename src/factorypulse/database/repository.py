from __future__ import annotations

from datetime import datetime

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

TABLE_NAME = "sensor_records"


def write_sensor_data(
    df: pd.DataFrame,
    engine: Engine,
    if_exists: str = "replace",
    chunksize: int = 10000,
) -> None:
    df.to_sql(
        TABLE_NAME,
        con=engine,
        if_exists=if_exists,
        index=False,
        method="multi",
        chunksize=chunksize,
    )


def read_sensor_data(
    engine: Engine,
    machine_id: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    query = """
    SELECT *
    FROM sensor_records
    WHERE 1=1
    """
    params: dict[str, object] = {}

    if machine_id is not None:
        query += " AND machine_id = :machine_id"
        params["machine_id"] = machine_id

    if start_time is not None:
        query += " AND timestamp >= :start_time"
        params["start_time"] = start_time

    if end_time is not None:
        query += " AND timestamp <= :end_time"
        params["end_time"] = end_time

    query += " ORDER BY timestamp"

    if limit is not None:
        query += " LIMIT :limit"
        params["limit"] = limit

    return pd.read_sql(text(query), con=engine, params=params)


def write_dataset_metadata(
    engine: Engine,
    dataset_name: str,
    dataset_version: str,
    source_path: str,
    row_count: int,
    anomaly_count: int,
    notes: str = "",
) -> None:
    query = text(
        """
        INSERT INTO dataset_metadata (
            dataset_name,
            dataset_version,
            created_at,
            source_path,
            row_count,
            anomaly_count,
            notes
        )
        VALUES (
            :dataset_name,
            :dataset_version,
            :created_at,
            :source_path,
            :row_count,
            :anomaly_count,
            :notes
        )
        """
    )

    with engine.begin() as conn:
        conn.execute(
            query,
            {
                "dataset_name": dataset_name,
                "dataset_version": dataset_version,
                "created_at": datetime.utcnow(),
                "source_path": source_path,
                "row_count": row_count,
                "anomaly_count": anomaly_count,
                "notes": notes,
            },
        )
