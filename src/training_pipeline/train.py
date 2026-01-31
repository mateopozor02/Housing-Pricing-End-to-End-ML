"""
Train a Light GBM model for regression tasks.

- Reads feature-engineered train/test datasets.
- Trains a Light GBM regressor.
- Returns metrics and saves model to 'model_output'
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

from pathlib import Path
from typing import Dict, Optional, Tuple
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor

DEFAULT_TRAIN_PATH = Path("data/processed/feature_engineered_train_data.csv")
DEFAULT_TEST_PATH = Path("data/processed/feature_engineered_test_data.csv")
DEFAULT_OUTPUT_PATH = Path("models/lightgbm_model.pkl")


def _maybe_sample(
    df: pd.DataFrame, sample_fraction: Optional[float], random_state: int
) -> pd.DataFrame:
    """Optionally sample the DataFrame for quicker training during tests."""
    if sample_fraction is None:
        return df
    sample_fraction = float(sample_fraction)
    if sample_fraction <= 0 or sample_fraction >= 1:
        return df
    return df.sample(frac=sample_fraction, random_state=random_state).reset_index(
        drop=True
    )


def train_model(
    train_path: Path | str = DEFAULT_TRAIN_PATH,
    test_path: Path | str = DEFAULT_TEST_PATH,
    model_output: Path | str = DEFAULT_OUTPUT_PATH,
    model_params: Optional[Dict] = None,
    sample_fraction: Optional[float] = None,
    random_state: int = 42,
) -> Tuple[LGBMRegressor, Dict[str, float]]:
    """
    Train a baseline LightGBM Model and save it.
    :return: Description
    :rtype: Tuple[LGBMRegressor, Dict[str, float]]
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Optionally sample the datasets
    train_df = _maybe_sample(train_df, sample_fraction, random_state)
    test_df = _maybe_sample(test_df, sample_fraction, random_state)

    # Convert datasets into LightGBM dataset objects and define target variable
    target = "price"
    train_data = lgb.Dataset(train_df.drop(columns=[target]), label=train_df[target])
    test_data = lgb.Dataset(test_df.drop(columns=[target]), label=test_df[target])

    params = {
        "objective": "regression",
        "metric": "mae,rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.15,
        "num_leaves": 31,
    }
    if model_params:
        params.update(model_params)

    num_round = 100

    # Train the model
    model = lgb.train(
        params,
        train_data,
        num_round,
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(stopping_rounds=5)],
    )

    # Calculate metrics
    y_pred = model.predict(
        test_df.drop(columns=["price"]), num_iteration=model.best_iteration
    )

    r2 = r2_score(test_df["price"], y_pred)
    rmse = np.sqrt(mean_squared_error(test_df["price"], y_pred))
    mae = mean_absolute_error(test_df["price"], y_pred)
    metrics = {"mae": mae, "rmse": rmse, "r2": r2}

    out = Path(model_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    dump(model, out)
    print(f"Model trained. Saved to {out}")
    print(f"   MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")

    return model, metrics


if __name__ == "__main__":
    train_model()
