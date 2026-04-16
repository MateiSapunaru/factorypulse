from __future__ import annotations

from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential

from factorypulse.azureml.job import load_azureml_config


def get_ml_client(config_path: str | Path) -> MLClient:
    config = load_azureml_config(config_path)
    return MLClient.from_config(
        credential=DefaultAzureCredential(),
        path=config.subscription_config_path,
    )


def main() -> None:
    # Replace this with your successful Azure ML job name
    job_name = "gifted_fork_50w4113kqq"

    ml_client = get_ml_client("configs/azureml.yaml")

    model_asset = Model(
        path=f"azureml://jobs/{job_name}/outputs/output_dir/paths/models",
        type=AssetTypes.CUSTOM_MODEL,
        name="factorypulse-lstm-autoencoder",
        description="FactoryPulse final LSTM autoencoder bundle with weights, scaler, and metadata.",
        tags={
            "project": "factorypulse",
            "stage": "final",
            "framework": "pytorch",
            "tracking": "azureml",
        },
    )

    registered_model = ml_client.models.create_or_update(model_asset)

    print("Model registered successfully.")
    print(f"Name: {registered_model.name}")
    print(f"Version: {registered_model.version}")
    print(f"ID: {registered_model.id}")


if __name__ == "__main__":
    main()