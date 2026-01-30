"""
Feature engineering module for the feature pipeline.
This module includes functions for creating and transforming features from preprocessed data.
Also, we save fitted encoders for inference time.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from category_encoders import TargetEncoder
from joblib import dump

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add separate date features to the dataframe from the 'date' column.
    Features added: year, month, quarter
    """
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter

    df.insert(1, "year", df.pop("year"))
    df.insert(2, "month", df.pop("month"))
    df.insert(3, "quarter", df.pop("quarter"))

    return df


def frequency_encode(
    train: pd.DataFrame, test: pd.DataFrame, column: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series[int]]:
    """
    Apply frequency encoding to a specified categorical column in train and test dataframes.
    """
    frequency_map = train[column].value_counts()

    train[column + "_freq"] = train[column].map(frequency_map)
    test[column + "_freq"] = test[column].map(frequency_map)

    return train, test, frequency_map


def target_encode(
    train: pd.DataFrame, test: pd.DataFrame, column: str, target: str
) -> tuple[pd.DataFrame, pd.DataFrame, TargetEncoder]:
    """
    Apply target encoding to a specified categorical column in train and test dataframes.
    """
    encoder = TargetEncoder(cols=[column])
    train[column + "_encoded"] = encoder.fit_transform(train[column], train[target])
    test[column + "_encoded"] = encoder.transform(test[column])

    return train, test, encoder


def drop_unused_columns(
    df: pd.DataFrame,
    columns: list[str] = ["date", "city_full", "city", "zipcode", "median_sale_price"],
) -> pd.DataFrame:
    """
    Drop specified unused columns from a dataframe.
    """
    df = df.drop(columns=[column for column in columns if column in df.columns])

    return df


# Run feature engineering pipeline
def run_feature_engineering_pipeline(
    train_path: Path | str | None = None,
    test_path: Path | str | None = None,
    holdout_path: Path | str | None = None,
    output_dir: Path | str = PROCESSED_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series[int], TargetEncoder]:
    """
    Run feature engineering pipeline on train, test, and holdout datasets.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Input paths
    if train_path is None:
        train_path = output_dir / "cleaning_train_data.csv"
    if test_path is None:
        test_path = output_dir / "cleaning_test_data.csv"
    if holdout_path is None:
        holdout_path = output_dir / "cleaning_holdout_data.csv"

    # Load datasets
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    holdout = pd.read_csv(holdout_path)

    # Add date features
    train = add_date_features(train)
    test = add_date_features(test)
    holdout = add_date_features(holdout)

    # Frequency encode 'zipcode' column
    frequency_map = None
    if "zipcode" in train.columns:
        train, test, frequency_map = frequency_encode(train, test, "zipcode")
        holdout["zipcode_freq"] = holdout["zipcode"].map(frequency_map).fillna(0)
        # Save frequency map
        dump(frequency_map, MODELS_DIR / "frequency_encoder.pkl")

    # Target encode 'city_full' column
    target_encoder = None
    if "city_full" in train.columns:
        train, test, target_encoder = target_encode(
            train, test, "city_full", target="median_list_price"
        )
        holdout["city_full_encoded"] = target_encoder.transform(holdout["city_full"])
        # Save target encoder
        dump(target_encoder, MODELS_DIR / "target_encoder.pkl")

    # Drop unused columns
    train = drop_unused_columns(train)
    test = drop_unused_columns(test)
    holdout = drop_unused_columns(holdout)

    # Save engineered datasets
    out_train_path = output_dir / "feature_engineered_train_data.csv"
    out_test_path = output_dir / "feature_engineered_test_data.csv"
    out_holdout_path = output_dir / "feature_engineered_holdout_data.csv"

    train.to_csv(out_train_path, index=False)
    test.to_csv(out_test_path, index=False)
    holdout.to_csv(out_holdout_path, index=False)

    return train, test, holdout, frequency_map, target_encoder
