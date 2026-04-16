
---

# FactoryPulse – Time Series Anomaly Detection

FactoryPulse is an end-to-end machine learning project focused on detecting anomalies in multivariate industrial time-series data. The system evaluates statistical, classical machine learning, and deep learning approaches, and integrates experiment tracking, cloud training, and an interactive dashboard.

---

## Overview

This project simulates a real-world industrial monitoring system and implements a full ML pipeline:

* Data generation and storage (PostgreSQL + Docker)
* Feature engineering on time-series data
* Baseline and advanced anomaly detection models
* Experiment tracking with MLflow
* Cloud training using Azure Machine Learning
* Interactive visualization via Streamlit dashboard

The goal is to understand how different approaches behave under realistic conditions, especially in **imbalanced anomaly detection scenarios**.

---

## Problem Definition

Industrial systems generate continuous streams of sensor data. Detecting anomalies in such data is challenging due to:

* Temporal dependencies
* Noise and variability
* Strong class imbalance (few anomalies)

The task is to identify abnormal patterns in machine behavior using:

* Temperature
* Vibration
* Pressure
* RPM
* Power draw
* Throughput

---

## Dataset

The dataset is synthetically generated to mimic industrial sensor behavior.

* ~190,000 observations
* Multiple machines
* Time-based splits (train / validation / test)
* Anomaly ratio: ~13%

Each record contains:

* Timestamp
* Machine ID
* Sensor readings
* Ground truth anomaly label

---

## Models

### Rolling Z-Score (Baseline)

A statistical approach using rolling mean and standard deviation per feature.

* Simple and interpretable
* Fails to capture temporal dependencies

---

### Isolation Forest

An unsupervised tree-based model for anomaly detection.

* Handles nonlinear relationships
* Sensitive to contamination parameter
* Produces many false positives

---

### LSTM Autoencoder

A sequence-based deep learning model trained to reconstruct normal behavior.

* Captures temporal structure in the data
* Uses reconstruction error for anomaly detection
* Provides the best overall performance

---

## Results

### Test Set Performance

| Model            | Precision | Recall | F1 Score | PR-AUC | FP / 1000 |
| ---------------- | --------- | ------ | -------- | ------ | --------- |
| Rolling Z-score  | 0.18      | 0.01   | 0.03     | 0.13   | ~8        |
| Isolation Forest | 0.13      | 0.98   | 0.23     | 0.15   | ~850      |
| LSTM Autoencoder | 0.27      | 0.72   | 0.39     | 0.50   | ~357      |

---

## Interpretation

* **Rolling Z-score** detects only extreme deviations → very low recall
* **Isolation Forest** overfits anomaly detection → extremely high false positives
* **LSTM Autoencoder** balances detection and noise → best overall model

The results highlight a common trade-off in anomaly detection:

> Increasing recall often leads to a large number of false positives.

---

## MLflow Tracking

All experiments are tracked using MLflow:

* Parameters
* Metrics (Precision, Recall, F1, PR-AUC)
* Artifacts:

  * Confusion matrices
  * Precision-recall curves
  * Time-series anomaly plots

### Screenshot – MLflow Experiment Tracking

*(Insert screenshot here)*

---

## Azure Machine Learning

The LSTM model is also trained in the cloud using Azure ML:

* Reproducible environments
* Remote job execution
* Integration with MLflow

---

## Dashboard

A Streamlit dashboard is included for interactive exploration:

* Dataset overview
* Time-series visualization
* Model comparison
* Inference demo

### Screenshot – Dataset Overview

*(Insert screenshot here)*

### Screenshot – Anomaly Visualization

*(Insert screenshot here)*

### Screenshot – Model Comparison

*(Insert screenshot here)*

### Screenshot – Inference Demo

*(Insert screenshot here)*

---

## Project Structure

```text
factorypulse/
│
├── data/
│   ├── processed/
│   └── artifacts/
│
├── src/factorypulse/
│   ├── models/
│   ├── training/
│   ├── inference/
│   └── database/
│
├── scripts/
│   ├── init_db.py
│   ├── load_to_postgres.py
│   └── submit_azureml_job.py
│
├── dashboard/
│   ├── app.py
│   └── pages/
│
├── configs/
│
└── README.md
```

---

## How to Run

### 1. Install dependencies

```bash
poetry install
```

---

### 2. Start PostgreSQL

```bash
docker compose up -d
```

---

### 3. Initialize database

```bash
poetry run python scripts/init_db.py
poetry run python scripts/load_to_postgres.py
```

---

### 4. Generate features

```bash
poetry run python scripts/run_training.py
```

---

### 5. Train models

```bash
poetry run python -m factorypulse.training.train_baseline
poetry run python -m factorypulse.training.train_lstm
```

---

### 6. Run dashboard

```bash
poetry run streamlit run dashboard/app.py
```

---

## Limitations

This project is an experimental system and not production-ready.

Key limitations:

* Synthetic data (no real industrial signals)
* Static anomaly thresholds
* High false positive rates
* No real-time streaming or deployment
* Limited model tuning

---

## Future Work

Potential improvements include:

* Dynamic thresholding
* Online anomaly detection
* Model monitoring and retraining
* Feature selection / dimensionality reduction
* Ensemble methods
* Transformer-based architectures

---

## Author

Matei Săpunaru
Machine Learning Engineer