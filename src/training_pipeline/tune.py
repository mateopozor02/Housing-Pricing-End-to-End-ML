"""
Hyperparameter tuning with Optuna + MLflow
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import mlflow

DEFAULT_TRAIN_PATH = Path("data/processed/feature_engineered_train_data.csv")
DEFAULT_TEST_PATH = Path("data/processed/feature_engineered_test_data.csv")
DEFAULT_OUTPUT_PATH = Path("models/lightgbm_best_model.pkl")


def _maybe_sample(
    df: pd.DataFrame, sample_fraction: Optional[float], random_state: int
) -> pd.DataFrame:
    """
    Optionally sample the DataFrame for quicker training during tests.
    """
    if sample_fraction is None:
        return df
    sample_fraction = float(sample_fraction)
    if sample_fraction <= 0 or sample_fraction >= 1:
        return df
    return df.sample(frac=sample_fraction, random_state=random_state).reset_index(
        drop=True
    )


def _load_data(
    train_path: Path | str,
    test_path: Path | str,
    sample_fraction: Optional[float],
    randon_state: int,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load datasets from given directories as path.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Optionally sample the datasets
    train_df = _maybe_sample(train_df, sample_fraction, randon_state)
    test_df = _maybe_sample(test_df, sample_fraction, randon_state)

    target = "price"
    X_train, y_train = train_df.drop(columns=[target]), train_df[target]
    X_test, y_test = test_df.drop(columns=[target]), test_df[target]

    return X_train, y_train, X_test, y_test


def tune_model(
    train_path: Path | str = DEFAULT_TRAIN_PATH,
    test_path: Path | str = DEFAULT_TEST_PATH,
    model_output: Path | str = DEFAULT_OUTPUT_PATH,
    n_trials: int = 20,
    sample_fraction: Optional[float] = None,
    tracking_uri: Optional[str] = None,
    experiment_name: str = "LightGBM_Hyperparameter_Tuning",
    random_state: int = 42,
) -> Tuple[Dict, Dict]:
    """
    Run optuna tuning; save best model; return (best_params, best_metrics)
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    X_train, y_train, X_test, y_test = _load_data(
        train_path, test_path, sample_fraction, random_state
    )

    def objective(trial: optuna.trial.Trial) -> float:
        """
        Objective function for Optuna hyperparameter tuning.

        Args:
            trial (optuna.trial.Trial): An Optuna trial object.

        Returns:
            float: The RMSE of the model on the test set.
        """
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 3000),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 10.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 10.0),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        }

        with mlflow.start_run():
            model = LGBMRegressor(**params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Log parameters and metrics to MLflow
            mlflow.log_params(params)
            mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})

        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    print("Best params from Optuna", best_params)

    # Re-train best model
    best_model = LGBMRegressor(**best_params)
    best_model.fit(X_train, y_train)

    # Evaluate best model
    y_pred = best_model.predict(X_test)
    best_metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }

    print("Best tuned model metrics:", best_metrics)

    # Save to models
    out = Path(model_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    dump(best_model, out)

    # Log final best model to MLflow
    with mlflow.start_run(run_name="Best-LightGBM-Model"):
        mlflow.log_params(best_params)
        mlflow.log_metrics(best_metrics)
        mlflow.lightgbm.log_model(best_model, name="model")

    return best_params, best_metrics


if __name__ == "__main__":
    tune_model()
