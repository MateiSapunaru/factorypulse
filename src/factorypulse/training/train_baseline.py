from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd
import yaml

from factorypulse.models.baseline.isolation_forest import (
    IsolationForestConfig,
    predict_isolation_forest,
    train_isolation_forest,
)
from factorypulse.models.baseline.rolling_zscore import (
    RollingZScoreConfig,
    compute_rolling_zscore_predictions,
)
from factorypulse.training.evaluate import (
    EvaluationResult,
    evaluate_binary_predictions,
    print_evaluation_result,
)
from factorypulse.training.mlflow_utils import (
    log_artifact_if_exists,
    log_metrics_with_prefix,
    log_params_flat,
    log_text_artifact,
    save_confusion_matrix_plot,
    save_metrics_summary_csv,
    save_pr_curve_plot,
    save_timeseries_anomaly_plot,
    setup_mlflow,
)


@dataclass(frozen=True)
class RollingZScoreSection:
    feature_columns: list[str]
    window_size: int
    z_threshold: float


@dataclass(frozen=True)
class IsolationForestSection:
    n_estimators: int
    contamination_values: list[float]
    random_state: int
    use_scaling: bool


@dataclass(frozen=True)
class MLflowSection:
    tracking_uri: str
    experiment_name: str
    run_name_prefix: str


@dataclass(frozen=True)
class BaselineSection:
    rolling_zscore: RollingZScoreSection
    isolation_forest: IsolationForestSection
    mlflow: MLflowSection


def load_baseline_config(config_path: str | Path) -> BaselineSection:
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    rz = RollingZScoreSection(**raw_config["baseline"]["rolling_zscore"])
    if_section = IsolationForestSection(**raw_config["baseline"]["isolation_forest"])
    mlflow_section = MLflowSection(**raw_config["mlflow"])

    return BaselineSection(
        rolling_zscore=rz,
        isolation_forest=if_section,
        mlflow=mlflow_section,
    )


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if col not in ["timestamp", "machine_id", "is_anomaly", "anomaly_type"]
    ]


def run_rolling_zscore(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: RollingZScoreSection,
) -> tuple[pd.DataFrame, EvaluationResult, EvaluationResult, EvaluationResult]:
    rz_config = RollingZScoreConfig(
        feature_columns=config.feature_columns,
        window_size=config.window_size,
        z_threshold=config.z_threshold,
        machine_id_column="machine_id",
    )

    train_pred = compute_rolling_zscore_predictions(train_df, rz_config)
    val_pred = compute_rolling_zscore_predictions(val_df, rz_config)
    test_pred = compute_rolling_zscore_predictions(test_df, rz_config)

    train_result = evaluate_binary_predictions(
        y_true=train_pred["is_anomaly"],
        y_pred=train_pred["prediction"],
        y_score=train_pred["anomaly_score"],
    )
    val_result = evaluate_binary_predictions(
        y_true=val_pred["is_anomaly"],
        y_pred=val_pred["prediction"],
        y_score=val_pred["anomaly_score"],
    )
    test_result = evaluate_binary_predictions(
        y_true=test_pred["is_anomaly"],
        y_pred=test_pred["prediction"],
        y_score=test_pred["anomaly_score"],
    )

    return test_pred, train_result, val_result, test_result


def run_isolation_forest(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: IsolationForestSection,
) -> tuple[pd.DataFrame, float, EvaluationResult, EvaluationResult, EvaluationResult]:
    feature_columns = select_feature_columns(train_df)

    best_model = None
    best_scaler = None
    best_val_pred = None
    best_test_pred = None
    best_contamination = None
    best_val_result = None

    print("\nIsolation Forest tuning on validation set:")

    for contamination in config.contamination_values:
        if_config = IsolationForestConfig(
            n_estimators=config.n_estimators,
            contamination=contamination,
            random_state=config.random_state,
            use_scaling=config.use_scaling,
        )

        model, scaler = train_isolation_forest(train_df, feature_columns, if_config)
        val_if = predict_isolation_forest(val_df, model, feature_columns, scaler)
        test_if = predict_isolation_forest(test_df, model, feature_columns, scaler)

        val_result_if = evaluate_binary_predictions(
            y_true=val_if["is_anomaly"],
            y_pred=val_if["prediction"],
            y_score=val_if["anomaly_score"],
        )

        print(
            f"contamination={contamination:.2f} | "
            f"precision={val_result_if.precision:.4f} | "
            f"recall={val_result_if.recall:.4f} | "
            f"f1={val_result_if.f1_score:.4f} | "
            f"pr_auc={val_result_if.pr_auc:.4f} | "
            f"fp/1000={val_result_if.false_positives_per_1000:.4f}"
        )

        if best_val_result is None or val_result_if.f1_score > best_val_result.f1_score:
            best_model = model
            best_scaler = scaler
            best_val_pred = val_if
            best_test_pred = test_if
            best_contamination = contamination
            best_val_result = val_result_if

    assert best_model is not None
    assert best_val_pred is not None
    assert best_test_pred is not None
    assert best_contamination is not None
    assert best_val_result is not None

    feature_columns = select_feature_columns(train_df)
    train_if = predict_isolation_forest(train_df, best_model, feature_columns, best_scaler)

    train_result_if = evaluate_binary_predictions(
        y_true=train_if["is_anomaly"],
        y_pred=train_if["prediction"],
        y_score=train_if["anomaly_score"],
    )
    test_result_if = evaluate_binary_predictions(
        y_true=best_test_pred["is_anomaly"],
        y_pred=best_test_pred["prediction"],
        y_score=best_test_pred["anomaly_score"],
    )

    return (
        best_test_pred,
        best_contamination,
        train_result_if,
        best_val_result,
        test_result_if,
    )


def main() -> None:
    config = load_baseline_config("configs/training.yaml")

    setup_mlflow(
        tracking_uri=config.mlflow.tracking_uri,
        experiment_name=config.mlflow.experiment_name,
    )

    train_df = pd.read_csv("data/processed/train_features.csv", parse_dates=["timestamp"])
    val_df = pd.read_csv("data/processed/val_features.csv", parse_dates=["timestamp"])
    test_df = pd.read_csv("data/processed/test_features.csv", parse_dates=["timestamp"])

    artifact_root = Path("data/artifacts")
    prediction_dir = artifact_root / "predictions"
    plot_dir = artifact_root / "plots"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Rolling Z-score run
    with mlflow.start_run(
        run_name=f"{config.mlflow.run_name_prefix}_rolling_zscore_{datetime.now():%Y%m%d_%H%M%S}"
    ):
        log_params_flat(
            {
                "model_name": "rolling_zscore",
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "test_rows": len(test_df),
            }
        )
        log_params_flat(config.rolling_zscore.__dict__, prefix="rolling_zscore")

        rz_test_pred, rz_train_result, rz_val_result, rz_test_result = run_rolling_zscore(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            config=config.rolling_zscore,
        )

        print_evaluation_result("Rolling Z-Score - Train", rz_train_result)
        print_evaluation_result("Rolling Z-Score - Validation", rz_val_result)
        print_evaluation_result("Rolling Z-Score - Test", rz_test_result)

        log_metrics_with_prefix(rz_train_result.to_dict(), prefix="train")
        log_metrics_with_prefix(rz_val_result.to_dict(), prefix="val")
        log_metrics_with_prefix(rz_test_result.to_dict(), prefix="test")

        rz_pred_path = prediction_dir / "rolling_zscore_test_predictions.csv"
        rz_test_pred.to_csv(rz_pred_path, index=False)
        log_artifact_if_exists(rz_pred_path)

        rz_cm_path = plot_dir / "rolling_zscore_confusion_matrix.png"
        save_confusion_matrix_plot(
            y_true=rz_test_pred["is_anomaly"],
            y_pred=rz_test_pred["prediction"],
            output_path=rz_cm_path,
            title="Rolling Z-Score - Test Confusion Matrix",
        )
        log_artifact_if_exists(rz_cm_path)

        rz_pr_path = plot_dir / "rolling_zscore_pr_curve.png"
        save_pr_curve_plot(
            y_true=rz_test_pred["is_anomaly"],
            y_score=rz_test_pred["anomaly_score"],
            output_path=rz_pr_path,
            title="Rolling Z-Score - Test PR Curve",
        )
        log_artifact_if_exists(rz_pr_path)

        rz_plot_df = rz_test_pred[rz_test_pred["machine_id"] == "machine_000"].copy()
        rz_ts_path = plot_dir / "rolling_zscore_timeseries_machine_000_temperature.png"
        save_timeseries_anomaly_plot(
            df=rz_plot_df,
            timestamp_column="timestamp",
            value_column="temperature",
            true_label_column="is_anomaly",
            pred_label_column="prediction",
            output_path=rz_ts_path,
            title="Rolling Z-Score - machine_000 - temperature",
        )
        log_artifact_if_exists(rz_ts_path)

        log_text_artifact(
            content="Rolling Z-score baseline using max per-feature absolute z-score.",
            artifact_file="notes/rolling_zscore_summary.txt",
        )

    # Isolation Forest run
    with mlflow.start_run(
        run_name=f"{config.mlflow.run_name_prefix}_isolation_forest_{datetime.now():%Y%m%d_%H%M%S}"
    ):
        log_params_flat(
            {
                "model_name": "isolation_forest",
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "test_rows": len(test_df),
            }
        )
        log_params_flat(
            {
                "n_estimators": config.isolation_forest.n_estimators,
                "random_state": config.isolation_forest.random_state,
                "use_scaling": config.isolation_forest.use_scaling,
                "contamination_values": ",".join(
                    str(v) for v in config.isolation_forest.contamination_values
                ),
            },
            prefix="isolation_forest",
        )

        (
            if_test_pred,
            best_contamination,
            if_train_result,
            if_val_result,
            if_test_result,
        ) = run_isolation_forest(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            config=config.isolation_forest,
        )

        print(f"\nBest Isolation Forest contamination: {best_contamination:.2f}")
        print_evaluation_result("Isolation Forest - Train", if_train_result)
        print_evaluation_result("Isolation Forest - Validation", if_val_result)
        print_evaluation_result("Isolation Forest - Test", if_test_result)

        mlflow.log_param("isolation_forest.best_contamination", best_contamination)
        log_metrics_with_prefix(if_train_result.to_dict(), prefix="train")
        log_metrics_with_prefix(if_val_result.to_dict(), prefix="val")
        log_metrics_with_prefix(if_test_result.to_dict(), prefix="test")

        if_pred_path = prediction_dir / "isolation_forest_test_predictions.csv"
        if_test_pred.to_csv(if_pred_path, index=False)
        log_artifact_if_exists(if_pred_path)

        if_cm_path = plot_dir / "isolation_forest_confusion_matrix.png"
        save_confusion_matrix_plot(
            y_true=if_test_pred["is_anomaly"],
            y_pred=if_test_pred["prediction"],
            output_path=if_cm_path,
            title="Isolation Forest - Test Confusion Matrix",
        )
        log_artifact_if_exists(if_cm_path)

        if_pr_path = plot_dir / "isolation_forest_pr_curve.png"
        save_pr_curve_plot(
            y_true=if_test_pred["is_anomaly"],
            y_score=if_test_pred["anomaly_score"],
            output_path=if_pr_path,
            title="Isolation Forest - Test PR Curve",
        )
        log_artifact_if_exists(if_pr_path)

        if_plot_df = if_test_pred[if_test_pred["machine_id"] == "machine_000"].copy()
        if_ts_path = plot_dir / "isolation_forest_timeseries_machine_000_temperature.png"
        save_timeseries_anomaly_plot(
            df=if_plot_df,
            timestamp_column="timestamp",
            value_column="temperature",
            true_label_column="is_anomaly",
            pred_label_column="prediction",
            output_path=if_ts_path,
            title="Isolation Forest - machine_000 - temperature",
        )
        log_artifact_if_exists(if_ts_path)

        log_text_artifact(
            content="Isolation Forest baseline tuned on validation F1 over contamination grid.",
            artifact_file="notes/isolation_forest_summary.txt",
        )

    summary_rows = [
        {
            "model": "rolling_zscore",
            "split": "train",
            "precision": rz_train_result.precision,
            "recall": rz_train_result.recall,
            "f1_score": rz_train_result.f1_score,
            "pr_auc": rz_train_result.pr_auc,
            "false_positives_per_1000": rz_train_result.false_positives_per_1000,
        },
        {
            "model": "rolling_zscore",
            "split": "validation",
            "precision": rz_val_result.precision,
            "recall": rz_val_result.recall,
            "f1_score": rz_val_result.f1_score,
            "pr_auc": rz_val_result.pr_auc,
            "false_positives_per_1000": rz_val_result.false_positives_per_1000,
        },
        {
            "model": "rolling_zscore",
            "split": "test",
            "precision": rz_test_result.precision,
            "recall": rz_test_result.recall,
            "f1_score": rz_test_result.f1_score,
            "pr_auc": rz_test_result.pr_auc,
            "false_positives_per_1000": rz_test_result.false_positives_per_1000,
        },
        {
            "model": "isolation_forest",
            "split": "train",
            "precision": if_train_result.precision,
            "recall": if_train_result.recall,
            "f1_score": if_train_result.f1_score,
            "pr_auc": if_train_result.pr_auc,
            "false_positives_per_1000": if_train_result.false_positives_per_1000,
        },
        {
            "model": "isolation_forest",
            "split": "validation",
            "precision": if_val_result.precision,
            "recall": if_val_result.recall,
            "f1_score": if_val_result.f1_score,
            "pr_auc": if_val_result.pr_auc,
            "false_positives_per_1000": if_val_result.false_positives_per_1000,
        },
        {
            "model": "isolation_forest",
            "split": "test",
            "precision": if_test_result.precision,
            "recall": if_test_result.recall,
            "f1_score": if_test_result.f1_score,
            "pr_auc": if_test_result.pr_auc,
            "false_positives_per_1000": if_test_result.false_positives_per_1000,
        },
    ]

    comparison_path = artifact_root / "model_comparison_summary.csv"
    save_metrics_summary_csv(summary_rows, comparison_path)

    best_model_name = (
        "isolation_forest"
        if if_val_result.f1_score > rz_val_result.f1_score
        else "rolling_zscore"
    )

    recommendation_text = f"""
FactoryPulse baseline comparison summary

Selected baseline: {best_model_name}

Selection rule:
- highest validation F1 score

Rolling Z-score validation F1: {rz_val_result.f1_score:.4f}
Isolation Forest validation F1: {if_val_result.f1_score:.4f}

Notes:
- Rolling Z-score is conservative and has low recall.
- Isolation Forest achieves much higher recall and better F1.
- Isolation Forest currently has high false positives and should be tuned further or improved with sequence models.
""".strip()

    recommendation_path = artifact_root / "baseline_recommendation.txt"
    recommendation_path.write_text(recommendation_text, encoding="utf-8")

    with mlflow.start_run(
        run_name=f"{config.mlflow.run_name_prefix}_comparison_{datetime.now():%Y%m%d_%H%M%S}"
    ):
        mlflow.log_param("run_type", "comparison_summary")
        mlflow.log_param("selected_baseline", best_model_name)

        log_artifact_if_exists(comparison_path)
        log_artifact_if_exists(recommendation_path)

        mlflow.log_metric("rolling_zscore_val_f1", rz_val_result.f1_score)
        mlflow.log_metric("isolation_forest_val_f1", if_val_result.f1_score)
        mlflow.log_metric("rolling_zscore_test_f1", rz_test_result.f1_score)
        mlflow.log_metric("isolation_forest_test_f1", if_test_result.f1_score)


if __name__ == "__main__":
    main()