"""
FastAPI application for serving housing price predictions.

This module provides a REST API for making predictions using the trained
LightGBM model. It accepts raw housing data and returns predicted prices.

Endpoints:
    POST /predict: Make predictions on input housing data
    GET /health: Health check endpoint
"""

from __future__ import annotations

import io
import boto3, os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import logging

from src.inference_pipeline.inference import predict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AWS S3 Configuration
S3_BUCKET = os.getenv("S3_BUCKET")
REGION = os.getenv("AWS_REGION")
S3_MODEL_KEY = "models/lightgbm_best_model.pkl"
S3_DATA_FE_KEY = "data/processed/feature_engineered_train_data.csv"

# Initialize S3 client
s3_client = boto3.client("s3", region_name=REGION)


def download_from_s3(bucket: str, key: str, local_path: Path) -> Path:
    """Download a file from S3 to local filesystem if it does not exist"""
    local_path = Path(local_path)
    if not local_path.exists():
        os.makedirs(local_path.parent, exist_ok=True)
        logger.info(f"Downloading {key} from S3...")
        s3_client.download_file(bucket, key, str(local_path))
    return str(local_path)


# Download model and training features if not cached
MODEL_PATH = Path(download_from_s3(S3_BUCKET, S3_MODEL_KEY, S3_MODEL_KEY))
TRAIN_FE_PATH = Path(download_from_s3(S3_BUCKET, S3_DATA_FE_KEY, S3_DATA_FE_KEY))

# Load expected training featured to ensure schema
if TRAIN_FE_PATH.exists():
    _train_columns = pd.read_csv(TRAIN_FE_PATH, nrows=1)
    TRAIN_FEATURE_COLUMNS = [
        column for column in _train_columns.columns if column != "price"
    ]
else:
    TRAIN_FEATURE_COLUMNS = None


# Initialize FastAPI app
app = FastAPI(
    title="Housing Price Prediction API",
    description="API for predicting housing prices using trained ML model",
    version="1.0.0",
)


@app.get("/")
async def root() -> Dict[str, str]:
    return {"message": "Housing Regression API running."}


@app.get("/health", tags=["health"])
def health_check() -> Dict[str, str]:
    """
    Health check endpoint to verify API is running.

    Returns:
        Dict with status message
    """
    status: Dict[str, Any] = {"model_path": str(MODEL_PATH)}
    if not MODEL_PATH.exists():
        status["status"] = "Unhealthy"
        status["error"] = "Model not found"
    else:
        status["status"] = "Healthy"
        if TRAIN_FEATURE_COLUMNS:
            status["n_features_expected"] = str(len(TRAIN_FEATURE_COLUMNS))
    return status


@app.post("/predict", tags=["predictions"])
async def predict_prices(data: List[dict]) -> Dict[str, Any]:
    if not MODEL_PATH.exists():
        return {"error": f"Model not found at {MODEL_PATH}"}

    df = pd.DataFrame(data)
    if df.empty:
        return {"error": "No data provided"}

    preds_df = predict(df, model_path=MODEL_PATH)
    response = {"predictions": preds_df["predicted_price"].astype(float).tolist()}
    if "actual_price" in preds_df.columns:
        response["actuals"] = preds_df["actual_price"].astype(float).tolist()

    return response
