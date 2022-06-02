from typing import Any, Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, make_scorer


def get_all_metrics() -> Dict[str, Any]:
    """
    Note: The weighted f1 score in sklearn is a bit weird. Weighting usually helps the minority classes, but in this
    case it actually helps the majority classes dominate.

    Conversely, Macro F1 penalizes models that neglect the minority classes.
    """
    metrics = {
        "weighted_f1": compute_weighed_f1,
        "macro_f1": compute_macro_f1,
        "accuracy": accuracy_score
    }
    return metrics


def macro_f1_scorer():
    return make_scorer(compute_macro_f1, greater_is_better=True)


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    macro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
    return macro_f1


def compute_weighed_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    weighed_f1 = f1_score(y_true=y_true, y_pred=y_pred, average="weighted")
    return weighed_f1
