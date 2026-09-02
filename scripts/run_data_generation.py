from __future__ import annotations

from pathlib import Path

from factorypulse.data_generation.generator import generate_base_dataset, load_generator_config


def main() -> None:
    config_path = Path("configs/data_generation.yaml")
    output_path = Path("data/raw/synthetic_data.csv")

    config = load_generator_config(config_path)
    df = generate_base_dataset(config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved dataset to: {output_path}")
    print(df.head())
    print(f"Shape: {df.shape}")
    print("\nAnomaly counts:")
    print(df["anomaly_type"].value_counts())


if __name__ == "__main__":
    main()
