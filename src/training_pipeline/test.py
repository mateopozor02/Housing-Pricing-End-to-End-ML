"""
Evaluate a LightGBM model on the test split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

from pathlib import Path
from typing import Dict, Optional, Tuple
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor

DEFAULT_TEST_PATH = Path("data/processed/feature_engineered_test_data.csv")
DEFAULT_MODEL_PATH = Path("models/lightgbm_model.pkl")


# TO DO: move this to a utils file. Also used in train.py
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


def evaluate_model(
    model_path: Path | str = DEFAULT_MODEL_PATH,
    test_path: Path | str = DEFAULT_TEST_PATH,
    sample_fraction: Optional[float] = None,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Evaluates a lightgbm model saved in models.
    :return: Description
    :rtype: Dict[str, float]
    """
    test_df = pd.read_csv(test_path)
    test_df = _maybe_sample(test_df, sample_fraction, random_state)

    # Convert datasets into LightGBM dataset objects and define target variable
    target = "price"
    test_data = lgb.Dataset(test_df.drop(columns=[target]), label=test_df[target])

    # Load model and predict
    model = load(model_path)
    y_pred = model.predict(
        test_df.drop(columns=["price"]), num_iteration=model.best_iteration
    )

    r2 = r2_score(test_df["price"], y_pred)
    rmse = np.sqrt(mean_squared_error(test_df["price"], y_pred))
    mae = mean_absolute_error(test_df["price"], y_pred)
    metrics = {"mae": mae, "rmse": rmse, "r2": r2}

    print("Evaluation:")
    print(f"   MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")
    return metrics


if __name__ == "__main__":
    evaluate_model()
