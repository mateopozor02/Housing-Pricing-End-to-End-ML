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

### Training Pipeline Structure

The project follows a **training pipeline** architecture in `src/training_pipeline/`:

1. **train.py**: Model training and hyperparameter tuning
   - Trains LightGBM regressors with customizable hyperparameters
   - Loads feature-engineered train/test datasets
   - Evaluates performance on test split during training
   - Saves trained model to `models/` directory as .pkl file
   - Returns trained model and evaluation metrics (MAE, RMSE, R²)
   - Key function: `train_model()` with optional `sample_fraction` for quick iteration
   - Uses `_maybe_sample()` utility function to optionally subsample data

2. **test.py**: Model evaluation on test split
   - Evaluates a trained LightGBM model on the test dataset
   - Computes MAE, RMSE, and R² metrics
   - Supports optional sampling via `sample_fraction` for validation during development
   - Prints formatted evaluation results to console
   - Key function: `evaluate_model()` loads model from disk and evaluates
   - Uses `_maybe_sample()` utility function to optionally subsample test data

3. **tune.py**: Hyperparameter optimization with Optuna
   - Performs Optuna-based hyperparameter search for LightGBM
   - Logs experiments with MLFlow for tracking and comparison
   - Searches over key parameters: num_leaves, learning_rate, lambda_l1, lambda_l2, etc.

### Data Flow

```
raw_housing_data.csv
  → load.py → {train,test,holdout}_data.csv
  → preprocess.py → cleaning_{train,test,holdout}_data.csv  
  → feature_engineering.py → feature_engineered_{train,test,holdout}_data.csv
  → train.py → models/lightgbm_model.pkl (trained on feature_engineered_train_data.csv)
  → test.py → evaluation metrics (evaluated on feature_engineered_test_data.csv)
```

### Model Training & Experiment Tracking

- **MLFlow** is used for experiment tracking (logs stored in `mlruns/` and `mlflow.db`)
- **Optuna** is used for hyperparameter tuning via `tune.py`
- Primary model: **LightGBM** (chosen over XGBoost based on MAE, RMSE, R2 performance)
- Models and encoders are saved to `models/` directory as .pkl files
- Target variable: `median_list_price` (referred to as `price` in training/evaluation)
- Evaluation metrics: MAE, RMSE, R²

### Directory Structure

```
data/
  raw/          # Initial splits from load.py
  processed/    # Cleaned and feature-engineered data
src/
  feature_pipeline/
    __init__.py
    load.py
    preprocess.py
    feature_engineering.py
  training_pipeline/
    __init__.py
    train.py
    test.py
    tune.py
models/         # Saved encoders and models (.pkl files)
notebooks/      # Jupyter notebooks for exploration and experimentation
tests/          # Test directory with comprehensive test coverage
  conftest.py
  test_load.py
  test_preprocess.py
  test_feature_engineering.py
  test_training_pipeline.py
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

## Running the Training Pipeline


```bash
# Step 1: Train model with default parameters
uv run python src/training_pipeline/train.py

# Step 2: Evaluate model on test split
uv run python src/training_pipeline/test.py

# Step 3 (Optional): Hyperparameter tuning with Optuna
uv run python src/training_pipeline/tune.py
```

### Running the Inference Pipeline

To run the inference pipeline, ensure you are executing from the project root and that Python can find the `src` package. If you encounter `No module named src`, set the `PYTHONPATH` to the project root:

```bash
PYTHONPATH=. uv run python src/inference_pipeline/inference.py --input <raw_csv> --output <predictions_csv>
```

This ensures absolute imports like `from src.feature_pipeline.preprocess import ...` work correctly. The inference pipeline will preprocess, feature engineer, align columns, and generate predictions using the trained model.

**Note:** Always use absolute imports and run scripts from the project root. If you use a different working directory, set `PYTHONPATH` accordingly.

Or run individual functions programmatically:

```python
from src.training_pipeline.train import train_model
from src.training_pipeline.test import evaluate_model

# Train model
model, train_metrics = train_model()
print(f"Training metrics: {train_metrics}")

# Evaluate on test set
eval_metrics = evaluate_model()
print(f"Evaluation metrics: {eval_metrics}")
```

## Key Technical Details

### Training Pipeline Functions

**`train_model()` - Training Function**
- Parameters:
  - `train_path`: Path to feature-engineered training data (default: `data/processed/feature_engineered_train_data.csv`)
  - `test_path`: Path to feature-engineered test data (default: `data/processed/feature_engineered_test_data.csv`)
  - `model_output`: Path to save trained model (default: `models/lightgbm_model.pkl`)
  - `model_params`: Dict of custom LightGBM hyperparameters (optional)
  - `sample_fraction`: Float between 0-1 to subsample data for quick iteration (optional)
  - `random_state`: Random seed for reproducibility (default: 42)
- Returns: Tuple of (trained LGBMRegressor model, metrics dict with "mae", "rmse", "r2")

**`evaluate_model()` - Evaluation Function**
- Parameters:
  - `model_path`: Path to saved LightGBM model (default: `models/lightgbm_model.pkl`)
  - `test_path`: Path to feature-engineered test data (default: `data/processed/feature_engineered_test_data.csv`)
  - `sample_fraction`: Float between 0-1 to subsample test data (optional)
  - `random_state`: Random seed for reproducibility (default: 42)
- Returns: Dict with keys "mae", "rmse", "r2"
- Side effect: Prints formatted metrics to console

**`_maybe_sample()` - Utility Function (both modules)**
- Used internally by both `train.py` and `test.py`
- Parameters:
  - `df`: DataFrame to optionally sample
  - `sample_fraction`: Float between 0-1, or None/0/>1 to return full data
  - `random_state`: Random seed for reproducibility
- Returns: Sampled DataFrame or full DataFrame with reset index

### Target Variable Naming

- Raw data: `median_list_price`
- After feature engineering and in training/evaluation: renamed to `price`
- Training pipeline expects `price` column in feature-engineered datasets

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
- **Testing**: pytest, pytest-cov

## Testing

The project uses **pytest** as the testing framework with comprehensive unit and integration tests maintaining 96% code coverage.

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

# Run tests with verbose output
uv run pytest -v

# Run with coverage report and specific marker
uv run pytest -m unit --cov=src --cov-report=html
```

### Test Structure

- `tests/conftest.py`: Shared fixtures for all tests
- `tests/test_load.py`: Tests for data loading and splitting
- `tests/test_preprocess.py`: Tests for data preprocessing
- `tests/test_feature_engineering.py`: Tests for feature engineering
- `tests/test_training_pipeline.py`: Comprehensive tests for training and evaluation modules

### Test Conventions

- Use `@pytest.mark.unit` for unit tests
- Use `@pytest.mark.integration` for integration tests
- Organize tests into classes (e.g., `TestFunctionName`)
- Use descriptive test method names (e.g., `test_returns_correct_value_when_input_is_valid`)
- Use fixtures from `conftest.py` for common test data
- Use temporary directories (`tmp_path`) for file I/O tests
- Mock external dependencies (file loading, model prediction)
- Include both positive and negative test cases
- Test edge cases (empty data, extreme values, etc.)

### Test Coverage

The project maintains 96% test coverage. After running tests with coverage, view the HTML report:

```bash
open htmlcov/index.html  # macOS
```

### Training Pipeline Tests

`tests/test_training_pipeline.py` includes:

**Unit Tests**
- `TestMaybeSample`: Tests for `_maybe_sample()` utility function
  - Sample fraction handling (None, 0, 1, 0.5, 0.25)
  - Reproducibility with random_state
  - Index reset and column preservation
  - Edge cases (negative fractions, fractions > 1)
  
- `TestTrainModel`: Tests for `train_model()` function
  - Returns model and metrics dict
  - Metrics are correct types and non-negative
  - Model saved to disk
  - Sample fraction parameter respected
  - Custom model parameters accepted
  
- `TestEvaluateModel`: Tests for `evaluate_model()` function
  - Returns metrics dict with correct keys
  - Metrics are float types and non-negative
  - Best iteration parameter used correctly
  - Price column dropped from features
  - Sample fraction support

**Integration Tests**
- Full training pipeline end-to-end
- Model reloading and evaluation
- Training with sample fractions for quick iteration
- Perfect predictions (R² ≈ 1.0)
- Random baseline predictions

## Notes for Development

- The project uses `__init__.py` files to structure packages (src, src/feature_pipeline, src/training_pipeline, tests)
- Always use `uv` instead of pip for dependency management
- MLFlow tracking URI defaults to local `mlruns/` directory
- Encoders must be saved during training and loaded during inference for consistency
- The target column name varies: `median_list_price` in raw data, `price` in training/evaluation
- Use relative imports within the src package structure: `from src.training_pipeline.train import train_model`
- All tests should pass before committing changes
- Models are saved as `.pkl` files using joblib for easy serialization/deserialization
