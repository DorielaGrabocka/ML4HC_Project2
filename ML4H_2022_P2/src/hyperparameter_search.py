
import datetime
import os
import time
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from src.constants import PATH_HYPERPARAMETER_SEARCH_RUNS
from src.metrics import macro_f1_scorer


def run_randomized_search(estimator, estimator_name, x, y, distributions: Dict[str, Any],
                          n_iter: int, cv: int, random_state: int, n_jobs=4) -> Tuple[pd.DataFrame, Dict[str, Any], float]:
    print("\nRunning hyperparameter search!\n")
    search = RandomizedSearchCV(estimator, distributions, n_iter=n_iter, cv=cv, scoring=macro_f1_scorer(), n_jobs=n_jobs,
                                random_state=random_state, verbose=3)
    search = search.fit(x, y)
    best_params, best_score = search.best_params_, search.best_score_
    print(f"Hyperparameter search finished! Best result: {best_score}\n")

    results = pd.DataFrame(search.cv_results_).sort_values(by="rank_test_score")
    sanity_check_best_results(best_params, best_score, results)
    results.to_csv(get_hyper_search_file_path(estimator_name))

    return results, best_params, best_score  # type: ignore[no-any-return]


def sanity_check_best_results(best_params: Dict, best_score: float, results: pd.DataFrame) -> None:
    assert results["mean_test_score"].iloc[0] == best_score

    # possibly multiple equally well performing setups
    best_params_list = results[results["mean_test_score"] == best_score]["params"].tolist()
    assert best_params in best_params_list


def get_hyper_search_file_path(clf_name: str) -> str:
    dir_path = PATH_HYPERPARAMETER_SEARCH_RUNS
    # ensure dir exists
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return dir_path + get_hyperparams_search_filename(clf_name)  # type: ignore[no-any-return]


def get_hyperparams_search_filename(clf_name: str) -> str:
    return clf_name + "_" + get_current_timestamp() + "_.csv"


def get_current_timestamp() -> str:
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')
