from __future__ import annotations

from factorypulse.database.connection import get_engine
from factorypulse.database.models import Base


def main() -> None:
    engine = get_engine()
    Base.metadata.create_all(engine)
    print("Database tables created successfully.")


if __name__ == "__main__":
    main()