# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is an end-to-end machine learning project for housing price prediction using the Kaggle housets dataset. The project uses **uv** as the package manager and follows a pipeline-based architecture for data processing, feature engineering, and model training.

## Project Components

The project consists of three main components:

1. **Feature & Training Pipeline** (`src/feature_pipeline/` and `src/training_pipeline/`): Data processing, feature engineering, and model training
2. **FastAPI Backend** (`src/api/main.py`): REST API for serving predictions
3. **Streamlit Frontend** (`streamlit_app.py`): Web UI for exploring predictions on holdout data

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

### API Architecture

The project includes a **FastAPI backend** (`src/api/main.py`) for serving predictions:

- **Endpoints**:
  - `GET /`: Root endpoint with status message
  - `GET /health`: Health check with model and feature information
  - `POST /predict`: Batch prediction endpoint accepting list of housing records
  
- **Features**:
  - Downloads model and training features from AWS S3 if not cached locally
  - Validates input data against training schema
  - Returns predictions with optional actual prices if available

### Frontend Architecture

The project includes a **Streamlit UI** (`streamlit_app.py`) for interactive exploration:

- **Features**:
  - Loads feature-engineered holdout data from AWS S3 (cached locally)
  - Interactive filters for year, month, and region
  - Real-time predictions via FastAPI backend
  - Performance metrics: MAE, RMSE, Avg % Error, R²
  - Visualizations: Time series trends, scatter plots of actual vs predicted
  - Data tables with predictions and actuals

### Data Flow

```
raw_housing_data.csv
  → load.py → {train,test,holdout}_data.csv
  → preprocess.py → cleaning_{train,test,holdout}_data.csv  
  → feature_engineering.py → feature_engineered_{train,test,holdout}_data.csv
  → train.py → models/lightgbm_model.pkl (trained on feature_engineered_train_data.csv)
  → test.py → evaluation metrics (evaluated on feature_engineered_test_data.csv)
  → FastAPI backend → predictions
  → Streamlit frontend → visualizations & exploration
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
  api/
    main.py     # FastAPI backend application
  feature_pipeline/
    __init__.py
    load.py
    preprocess.py
    feature_engineering.py
  inference_pipeline/
    __init__.py
    inference.py
  training_pipeline/
    __init__.py
    train.py
    test.py
    tune.py
scripts/
    s3_upload.py        
models/         # Saved encoders and models (.pkl files)
notebooks/      # Jupyter notebooks for exploration and experimentation
tests/          # Test directory with comprehensive test coverage
  conftest.py
  test_load.py
  test_preprocess.py
  test_feature_engineering.py
  test_training_pipeline.py
mlruns/         # MLFlow experiment tracking
streamlit_app.py  # Streamlit frontend application
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

## Running the Backend & Frontend Locally

### Prerequisites

1. Ensure all dependencies are installed:
   ```bash
   uv sync
   ```

2. Create a `.env` file in the project root with required environment variables:
   ```bash
   # AWS S3 Configuration (for downloading model and data)
   S3_BUCKET=housing-pricing-regression-data
   AWS_REGION=us-east-2
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   
   # API Configuration
   API_URL=http://127.0.0.1:8000/predict
   
   # Optional MLFlow Configuration
   MLFLOW_TRACKING_URI=http://127.0.0.1:5000
   ```

3. Ensure the trained model exists:
   - Option 1: Run `uv run python src/training_pipeline/train.py` to train locally
   - Option 2: Download pre-trained model from `models/lightgbm_best_model.pkl`

### Option 1: Run Locally with Local Model

If you have a trained model in `models/lightgbm_best_model.pkl`:

**Terminal 1 - Start FastAPI Backend:**
```bash
uv run python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at `http://127.0.0.1:8000` with interactive docs at `http://127.0.0.1:8000/docs`

**Terminal 2 - Start Streamlit Frontend:**
```bash
uv run streamlit run streamlit_app.py
```

The UI will be available at `http://localhost:8501`

### Option 2: Run with AWS S3 Integration

If you want to download model and data from AWS S3:

**Terminal 1 - Start FastAPI Backend (downloads from S3):**
```bash
# Ensure AWS credentials are set in .env or AWS CLI
uv run python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

The backend will automatically download:
- `models/lightgbm_best_model.pkl` from S3 if not cached locally
- `data/processed/feature_engineered_train_data.csv` from S3 if not cached locally

**Terminal 2 - Start Streamlit Frontend:**
```bash
uv run streamlit run streamlit_app.py
```

The frontend will automatically download holdout dataset files from S3.

### Testing the API

**Health Check:**
```bash
curl http://127.0.0.1:8000/health
```

Expected response:
```json
{
  "model_path": "models/lightgbm_best_model.pkl",
  "status": "Healthy",
  "n_features_expected": "39"
}
```

**Make Predictions:**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '[{
    "date": "2023-01-01",
    "median_sale_price": 300000,
    "median_list_price": 320000,
    ...other features...
  }]'
```

### Stopping the Services

```bash
# Terminate both processes
# Terminal 1: Press Ctrl+C
# Terminal 2: Press Ctrl+C
```

## Running the Inference Pipeline

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

### API Functions

**`predict()` - Inference Function** (from `src/inference_pipeline/inference.py`)
- Parameters:
  - `input_df`: Raw housing data DataFrame
  - `model_path`: Path to trained LightGBM model (default: `models/lightgbm_best_model.pkl`)
  - `freq_encoder_path`: Path to frequency encoder (default: `models/frequency_encoder.pkl`)
  - `target_encoder_path`: Path to target encoder (default: `models/target_encoder.pkl`)
- Returns: DataFrame with `predicted_price` column and optionally `actual_price` if available
- Performs full pipeline: preprocessing → feature engineering → prediction

**`/predict` - REST Endpoint**
- Method: `POST`
- Input: List of dictionaries with housing features
- Returns: JSON with `predictions` array and optional `actuals` array
- Example:
  ```bash
  curl -X POST http://127.0.0.1:8000/predict \
    -H "Content-Type: application/json" \
    -d '[{"feature1": value1, "feature2": value2, ...}]'
  ```

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
- **Web Framework**: FastAPI, uvicorn
- **Frontend**: streamlit, plotly
- **Cloud**: boto3 (AWS S3)
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

## Troubleshooting

### Backend Issues

**"No module named src" when running API:**
```bash
# Set PYTHONPATH to project root
PYTHONPATH=. uv run python -m uvicorn src.api.main:app --reload
```

**"Model not found" error:**
- Check if `models/lightgbm_best_model.pkl` exists
- If using S3, verify AWS credentials in `.env`
- Train a new model: `uv run python src/training_pipeline/train.py`

**Port 8000 already in use:**
```bash
# Use a different port
uv run python -m uvicorn src.api.main:app --port 8001
```

### Frontend Issues

**"Could not connect to API server":**
- Ensure FastAPI backend is running on correct port (default: 8000)
- Check `API_URL` environment variable in Streamlit
- Update in `streamlit_app.py` if needed

**"S3 credentials not found":**
- Set AWS credentials in `.env` or AWS CLI
- For local testing without S3, ensure files are cached locally

**Data download timeout:**
- Increase timeout in `streamlit_app.py` line 43-50

## Notes for Development

- The project uses `__init__.py` files to structure packages (src, src/feature_pipeline, src/training_pipeline, tests)
- Always use `uv` instead of pip for dependency management
- MLFlow tracking URI defaults to local `mlruns/` directory
- Encoders must be saved during training and loaded during inference for consistency
- The target column name varies: `median_list_price` in raw data, `price` in training/evaluation
- Use absolute imports within the src package: `from src.training_pipeline.train import train_model`
- Run scripts from the project root to ensure correct relative paths
- All tests should pass before committing changes
- Models are saved as `.pkl` files using joblib for easy serialization/deserialization
- FastAPI backend and Streamlit frontend run independently; ensure they communicate via the API URL
