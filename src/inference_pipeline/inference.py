"""
Inference Pipeline Module

This module implements the inference pipeline for housing price predictions.

The inference pipeline follows these steps:
1. **Raw Input**: Accepts raw housing data with features in their original form
2. **Preprocessing**: Cleans and transforms raw data (handling missing values, encoding categorical variables, etc.)
3. **Feature Engineering**: Creates new features and applies transformations to match the training data distribution
4. **Feature Alignment**: Ensures features are aligned with the training dataset (same names, order, and types)
5. **Predictions**: Passes processed features through the trained model to generate housing price predictions

The pipeline ensures consistency between training and inference by reusing the same preprocessing
and feature engineering steps, maintaining reproducibility and reducing data drift.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from joblib import load

# Import helpers from feature pipeline
from src.feature_pipeline.preprocess import (
    clean_and_merge_city_names,
    remove_duplicatates,
    remove_outliers,
)
from src.feature_pipeline.feature_engineering import (
    add_date_features,
    drop_unused_columns,
)

# Set default paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL = PROJECT_ROOT / "models" / "lightgbm_best_model.pkl"
DEFAULT_FREQ_ENCODER = PROJECT_ROOT / "models" / "frequency_encoder.pkl"
DEFAULT_TARGET_ENCODER = PROJECT_ROOT / "models" / "target_encoder.pkl"
TRAIN_FE_PATH = (
    PROJECT_ROOT / "data" / "processed" / "feature_engineered_train_data.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "predictions.csv"

print("Inference using root:", PROJECT_ROOT)

# Load trianing feature columns
if TRAIN_FE_PATH.exists():
    _train_cols = pd.read_csv(TRAIN_FE_PATH, nrows=1)
    TRAIN_FEATURE_COLUMNS = [column for column in _train_cols if column != "price"]
else:
    TRAIN_FEATURE_COLUMNS = None


# Inference function
def predict(
    input_df: pd.DataFrame,
    model_path: Path | str = DEFAULT_MODEL,
    freq_encoder_path: Path | str = DEFAULT_FREQ_ENCODER,
    target_encoder_path: Path | str = DEFAULT_TARGET_ENCODER,
) -> pd.DataFrame:
    # Process raw input
    df = clean_and_merge_city_names(input_df)
    df = drop_unused_columns(df)
    df = remove_outliers(df)

    # Feature engineering
    if "date" in df.columns:
        df = add_date_features(df)

    # Encodings
    if Path(freq_encoder_path).exists() and "zipcode" in df.columns:
        freq_map = load(freq_encoder_path)
        df["zipcode_freq"] = df["zipcode"].map(freq_map).fillna(0)
        df = df.drop(columns=["zipcode"], errors="ignore")

    if Path(target_encoder_path).exists() and "city_full" in df.columns:
        target_encoder = load(target_encoder_path)
        df["city_full_encoded"] = target_encoder.transform(df["city_full"])
        df = df.drop(columns=["city_full"], errors="ignore")

    # Drop unused columns
    df = drop_unused_columns(df.copy())

    # Separate target if present
    y_true = None
    if "price" in df.columns:
        y_true = df["price"].tolist()
        df = df.drop(columns=["price"])

    # Align columns with training schema
    if TRAIN_FEATURE_COLUMNS is not None:
        df = df.reindex(columns=TRAIN_FEATURE_COLUMNS, fill_value=0)

    # Load model and predict
    model = load(model_path)
    predictions = model.predict(df)

    # Build outputs
    out = df.copy()
    out["predicted_price"] = predictions
    if y_true is not None:
        out["actual_price"] = y_true
        # Calculate r2 error
        from sklearn.metrics import r2_score

        r2 = r2_score(y_true, predictions)
        print(f"R2 error: {r2:.4f}")

    return out


# CLI entrypoint
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on new housing data (raw)."
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input RAW CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Path to save predictions CSV",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL),
        help="Path to trained model file",
    )
    parser.add_argument(
        "--freq_encoder",
        type=str,
        default=str(DEFAULT_FREQ_ENCODER),
        help="Path to frequency encoder pickle",
    )
    parser.add_argument(
        "--target_encoder",
        type=str,
        default=str(DEFAULT_TARGET_ENCODER),
        help="Path to target encoder pickle",
    )

    args = parser.parse_args()

    raw_df = pd.read_csv(args.input)
    preds_df = predict(
        raw_df,
        model_path=args.model,
        freq_encoder_path=args.freq_encoder,
        target_encoder_path=args.target_encoder,
    )

    preds_df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")
