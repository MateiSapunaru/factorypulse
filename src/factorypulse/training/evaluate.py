from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


@dataclass(frozen=True)
class EvaluationResult:
    precision: float
    recall: float
    f1_score: float
    pr_auc: float
    false_positives_per_1000: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def evaluate_binary_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_score: pd.Series | None = None,
) -> EvaluationResult:
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    if y_score is None:
        y_score = y_pred

    pr_auc = average_precision_score(y_true, y_score)

    false_positives = int(((y_true == 0) & (y_pred == 1)).sum())
    false_positives_per_1000 = false_positives / len(y_true) * 1000.0

    return EvaluationResult(
        precision=precision,
        recall=recall,
        f1_score=f1,
        pr_auc=pr_auc,
        false_positives_per_1000=false_positives_per_1000,
    )


def print_evaluation_result(model_name: str, result: EvaluationResult) -> None:
    print(f"\n{model_name}")
    print(f"Precision: {result.precision:.4f}")
    print(f"Recall:    {result.recall:.4f}")
    print(f"F1 Score:  {result.f1_score:.4f}")
    print(f"PR-AUC:    {result.pr_auc:.4f}")
    print(f"FP / 1000: {result.false_positives_per_1000:.4f}")
