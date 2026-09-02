from __future__ import annotations

from factorypulse.azureml.job import (
    build_lstm_command_job,
    create_environment,
    get_ml_client,
    load_azureml_config,
)


def main() -> None:
    config = load_azureml_config("configs/azureml.yaml")

    ml_client = get_ml_client(config.subscription_config_path)

    environment = create_environment(config)
    ml_client.environments.create_or_update(environment)
    print(f"Registered environment: " f"{config.environment.name}:{config.environment.version}")

    job = build_lstm_command_job(config, environment)
    returned_job = ml_client.jobs.create_or_update(job)

    print(f"Submitted Azure ML job: {returned_job.name}")
    print(f"Studio URL: {returned_job.studio_url}")


if __name__ == "__main__":
    main()
