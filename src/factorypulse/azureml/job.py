from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.entities import Command, Environment
from azure.identity import DefaultAzureCredential


@dataclass(frozen=True)
class JobSection:
    experiment_name: str
    display_name: str
    description: str
    compute_name: str


@dataclass(frozen=True)
class EnvironmentSection:
    name: str
    version: str
    base_image: str
    conda_file: str


@dataclass(frozen=True)
class InputsSection:
    processed_data_dir: str


@dataclass(frozen=True)
class OutputsSection:
    output_dir_name: str


@dataclass(frozen=True)
class AzureMLConfig:
    subscription_config_path: str
    job: JobSection
    environment: EnvironmentSection
    inputs: InputsSection
    outputs: OutputsSection


def load_azureml_config(config_path: str | Path) -> AzureMLConfig:
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return AzureMLConfig(
        subscription_config_path=raw["subscription_config_path"],
        job=JobSection(**raw["job"]),
        environment=EnvironmentSection(**raw["environment"]),
        inputs=InputsSection(**raw["inputs"]),
        outputs=OutputsSection(**raw["outputs"]),
    )


def get_ml_client(subscription_config_path: str | Path) -> MLClient:
    subscription_config_path = Path(subscription_config_path)
    credential = DefaultAzureCredential()
    return MLClient.from_config(credential=credential, path=subscription_config_path)


def create_environment(config: AzureMLConfig) -> Environment:
    return Environment(
        name=config.environment.name,
        version=config.environment.version,
        image=config.environment.base_image,
        conda_file=config.environment.conda_file,
        description="FactoryPulse LSTM Autoencoder environment",
    )


def build_lstm_command_job(config: AzureMLConfig, environment: Environment) -> Command:
    processed_data_path = Path(config.inputs.processed_data_dir).resolve()

    return command(
        code=".",
        command=(
            "PYTHONPATH=./src "
            "python -m factorypulse.training.train_lstm_azure "
            "--data_dir ${{inputs.processed_data}} "
            "--config_path configs/training.yaml "
            "--output_dir ${{outputs.output_dir}}"
        ),
        environment=environment,
        inputs={
            "processed_data": Input(
                type="uri_folder",
                path=str(processed_data_path),
            )
        },
        outputs={
            "output_dir": Output(
                type="uri_folder",
                mode="rw_mount",
            )
        },
        compute=config.job.compute_name,
        experiment_name=config.job.experiment_name,
        display_name=config.job.display_name,
        description=config.job.description,
    )
