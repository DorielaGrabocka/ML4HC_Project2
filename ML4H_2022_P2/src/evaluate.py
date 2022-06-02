import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from src.constants import LABELS
from src.metrics import get_all_metrics


def train_and_eval_clf(clf, x_train, y_train_labels, x_dev, y_dev_labels, x_test, y_test_labels, model_name):
    clf.fit(x_train, y_train_labels)

    print(f"Evaluating {model_name} performance on the dev set")
    predict_and_eval(clf, x_dev, y_dev_labels)

    print(f"\n\nEvaluating {model_name} performance on the test set")
    predict_and_eval(clf, x_test, y_test_labels)


def predict_and_eval(clf, x, y_labels) -> None:
    predicted = clf.predict(x)
    evaluate_predictions(y_labels, predicted)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print("Main metrics:")
    for k, v in get_all_metrics().items():
        perf = v(y_true, y_pred)
        print(k, perf)

    print(f"\nConfusion Matrix for label order: {LABELS}\n")
    print(confusion_matrix(y_true=y_true, y_pred=y_pred))

    print("\n\nDetailed classification report: ")
    print(classification_report(y_true, y_pred, target_names=LABELS))
