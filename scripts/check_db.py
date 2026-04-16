from __future__ import annotations

from factorypulse.database.connection import get_engine
from factorypulse.database.repository import read_sensor_data


def main() -> None:
    engine = get_engine()

    df = read_sensor_data(
        engine=engine,
        machine_id="machine_000",
        limit=5,
    )

    print(df)


if __name__ == "__main__":
    main()