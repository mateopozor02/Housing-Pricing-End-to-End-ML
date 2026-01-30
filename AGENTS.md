# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is an end-to-end machine learning project for housing price prediction using the Kaggle housets dataset. The project uses **uv** as the package manager and follows a pipeline-based architecture for data processing, feature engineering, and model training.

## Package Management

This project uses **uv** (not pip or poetry) for dependency management:

```bash
# Install dependencies
uv sync

# Add a new dependency
uv add <package-name>

# Run Python scripts
uv run python <script.py>

# Activate virtual environment
source .venv/bin/activate
```

## Project Architecture

### Data Pipeline Structure

The project follows a **feature pipeline** architecture in `src/feature_pipeline/`:

1. **load.py**: Time-based data splitting
   - Splits raw data into train/test/holdout based on dates
   - Train: before 2020-01-01
   - Test: 2020-01-01 to 2022-01-01  
   - Holdout: after 2022-01-01
   - Outputs to `data/raw/`

2. **preprocess.py**: Data cleaning and preprocessing
   - City name normalization using `CITY_MAPPING` for non-standard metro names
   - Merges with `raw_usmetros.csv` to add lat/lng coordinates
   - Removes duplicates (excluding date/year columns)
   - Outlier removal (median_list_price > 19M)
   - Outputs to `data/processed/` with "cleaning_" prefix

3. **feature_engineering.py**: Feature extraction and encoding
   - Date features: year, month, quarter extracted from date column
   - Frequency encoding for zipcode column
   - Target encoding for city_full column (using category_encoders.TargetEncoder)
   - Saves fitted encoders to `models/` directory as .pkl files for inference
   - Outputs to `data/processed/` with "feature_engineered_" prefix

### Data Flow

```
raw_housing_data.csv
  → load.py → {train,test,holdout}_data.csv
  → preprocess.py → cleaning_{train,test,holdout}_data.csv  
  → feature_engineering.py → feature_engineered_{train,test,holdout}_data.csv
```

### Model Training & Experiment Tracking

- **MLFlow** is used for experiment tracking (logs stored in `mlruns/` and `mlflow.db`)
- **Optuna** is used for hyperparameter tuning
- Primary model: **LightGBM** (chosen over XGBoost based on MAE, RMSE, R2 performance)
- Models and encoders are saved to `models/` directory
- Target variable: `median_list_price` (referred to as `price` in notebooks)
- Evaluation metrics: MAE, RMSE, R2

### Directory Structure

```
data/
  raw/          # Initial splits from load.py
  processed/    # Cleaned and feature-engineered data
src/
  feature_pipeline/
    load.py
    preprocess.py
    feature_engineering.py
models/         # Saved encoders and models (.pkl files)
notebooks/      # Jupyter notebooks for exploration and experimentation
tests/          # Test directory (currently empty)
mlruns/         # MLFlow experiment tracking
```

## Running the Feature Pipeline

```bash
# Step 1: Load and split raw data
uv run python src/feature_pipeline/load.py

# Step 2: Preprocess data splits
uv run python src/feature_pipeline/preprocess.py

# Step 3: Feature engineering
uv run python -c "from src.feature_pipeline.feature_engineering import run_feature_engineering_pipeline; run_feature_engineering_pipeline()"
```

Or run individual functions by importing them:

```python
from src.feature_pipeline.preprocess import run_preprocessing_pipeline
from src.feature_pipeline.feature_engineering import run_feature_engineering_pipeline

run_preprocessing_pipeline()
run_feature_engineering_pipeline()
```

## Key Technical Details

### Encoding Strategy

- **Frequency Encoding**: Applied to `zipcode` column to capture popularity of locations
- **Target Encoding**: Applied to `city_full` column using median_list_price as target
- Both encoders are fit on training data and saved as .pkl files for consistent inference

### Important Data Considerations

- The `CITY_MAPPING` dictionary in `preprocess.py` handles non-standard metro names (e.g., "DC_Metro" → "Washington-Arlington-Alexandria, DC-VA-MD-WV")
- City names are normalized (lowercase, standardized dashes, trimmed spaces)
- The `metros_path` parameter should point to `data/raw/raw_usmetros.csv` for lat/lng merging
- Duplicate removal uses all columns except "date" and "year"

### MLFlow Usage

When training models with MLFlow:

```python
import mlflow

# Set experiment
mlflow.set_experiment("experiment_name")

# Log runs
with mlflow.start_run():
    # ... training code ...
    mlflow.log_params(params)
    mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
    mlflow.lightgbm.log_model(model, "model")
```

## Dependencies

Key libraries used:
- **Data Processing**: pandas, numpy
- **ML Models**: lightgbm, xgboost, scikit-learn
- **Feature Engineering**: category-encoders
- **Experiment Tracking**: mlflow
- **Hyperparameter Tuning**: optuna
- **Data Quality**: great-expectations, evidently
- **Deployment**: fastapi, streamlit

## Testing

The project uses **pytest** as the testing framework with comprehensive unit and integration tests.

### Running Tests

```bash
# Run all tests
uv run pytest

# Run tests with coverage report
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/test_load.py

# Run only unit tests
uv run pytest -m unit

# Run only integration tests
uv run pytest -m integration
```

### Test Structure

- `tests/conftest.py`: Shared fixtures for all tests
- `tests/test_load.py`: Tests for data loading and splitting
- `tests/test_preprocess.py`: Tests for data preprocessing
- `tests/test_feature_engineering.py`: Tests for feature engineering

### Test Coverage

The project maintains 96% test coverage. After running tests with coverage, view the HTML report:

```bash
open htmlcov/index.html  # macOS
```

## Notes for Development

- The project uses `__init__.py` files to structure packages (src, src/feature_pipeline, tests)
- Always use `uv` instead of pip for dependency management
- MLFlow tracking URI defaults to local `mlruns/` directory
- Encoders must be saved during training and loaded during inference for consistency
- The target column name varies: `median_list_price` in raw data, sometimes `price` in notebooks
- All tests should pass before committing changes
