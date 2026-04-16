from __future__ import annotations

from sqlalchemy import Column, DateTime, Float, Index, Integer, String
from sqlalchemy.orm import declarative_base


Base = declarative_base()


class SensorRecord(Base):
    __tablename__ = "sensor_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)
    machine_id = Column(String(64), nullable=False)

    temperature = Column(Float, nullable=False)
    vibration = Column(Float, nullable=False)
    pressure = Column(Float, nullable=False)
    rpm = Column(Float, nullable=False)
    power_draw = Column(Float, nullable=False)
    throughput = Column(Float, nullable=False)

    is_anomaly = Column(Integer, nullable=False, default=0)
    anomaly_type = Column(String(64), nullable=False, default="none")

    __table_args__ = (
        Index("ix_sensor_records_timestamp", "timestamp"),
        Index("ix_sensor_records_machine_id", "machine_id"),
        Index("ix_sensor_records_machine_time", "machine_id", "timestamp"),
        Index("ix_sensor_records_is_anomaly", "is_anomaly"),
        Index("ix_sensor_records_anomaly_type", "anomaly_type"),
    )


class DatasetMetadata(Base):
    __tablename__ = "dataset_metadata"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_name = Column(String(128), nullable=False)
    dataset_version = Column(String(64), nullable=False)
    created_at = Column(DateTime, nullable=False)
    source_path = Column(String(512), nullable=False)
    row_count = Column(Integer, nullable=False)
    anomaly_count = Column(Integer, nullable=False)
    notes = Column(String(1024), nullable=False, default="")