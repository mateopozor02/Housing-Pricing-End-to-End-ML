# Housing Pricing Prediction End-to-End ML Project

A comprehensive machine learning solution for predicting housing prices using LightGBM, featuring end-to-end data processing, model training, containerization, and deployment. This project demonstrates production-grade ML engineering practices with data pipelines, experiment tracking, hyperparameter tuning, and RESTful API deployment.

**Dataset**: [Kaggle Housets Dataset](https://www.kaggle.com/datasets/shengkunwang/housets-dataset)

## Table of Contents

- [Model Overview](#model-overview)
- [Project Architecture](#project-architecture)
- [Feature Pipeline](#feature-pipeline)
- [Training Pipeline](#training-pipeline)
- [Containerization & Deployment](#containerization--deployment)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Running the Pipelines](#running-the-pipelines)
- [Testing](#testing)
- [Project Structure](#project-structure)

## Model Overview

### Model: LightGBM (Light Gradient Boosting Machine)

This project uses **LightGBM** as the primary regression model for housing price prediction. LightGBM was selected over alternative models (XGBoost, etc.) based on comprehensive performance evaluation on the Kaggle Housets dataset.

#### Why LightGBM?

1. **Superior Performance**: Achieves better Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² metrics compared to alternatives
2. **Computational Efficiency**: 
   - Faster training times due to leaf-wise tree growth strategy
   - Reduced memory footprint compared to depth-wise algorithms
   - Supports categorical features natively
3. **Production Ready**: Lightweight, fast inference for real-time predictions
4. **Scalability**: Efficient handling of large datasets with millions of rows
5. **Hyperparameter Flexibility**: Rich set of hyperparameters for fine-tuning via Optuna

#### Model Configuration

```python
{
    "objective": "regression",
    "metric": ["mae", "rmse"],
    "boosting_type": "gbdt",
    "learning_rate": 0.15,
    "num_leaves": 31,
    "num_rounds": 100
}
```

**Target Variable**: `median_list_price` (housing price in USD)

**Performance Metrics**:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

## Project Architecture

The project follows a modular, production-grade architecture with clear separation of concerns:

```
Raw Data
   ↓
Feature Pipeline (data/raw → data/processed)
   ├── Load & Split: Time-based train/test/holdout split
   ├── Preprocess: Cleaning, normalization, outlier removal
   └── Feature Engineering: Encoding, date features, target encoding
   ↓
Training Pipeline (models/)
   ├── Train: LightGBM model training with validation
   ├── Test: Evaluation on test set
   └── Tune: Hyperparameter optimization with Optuna
   ↓
Inference Pipeline (API/Frontend)
   ├── FastAPI Backend: /predict endpoint
   └── Streamlit Frontend: Interactive UI
```

### Key Components

1. **Feature Pipeline** (`src/feature_pipeline/`)
   - Data loading and temporal splitting
   - Preprocessing and outlier detection
   - Feature engineering with multiple encoding strategies

2. **Training Pipeline** (`src/training_pipeline/`)
   - Model training and evaluation
   - Experiment tracking with MLFlow
   - Hyperparameter tuning with Optuna

3. **Inference Pipeline** (`src/inference_pipeline/`)
   - Batch predictions
   - Consistent preprocessing for inference

4. **API Backend** (`src/api/main.py`)
   - FastAPI REST API
   - AWS S3 integration for model/data serving
   - Health checks and batch predictions

5. **Frontend** (`streamlit_app.py`)
   - Interactive exploration of predictions
   - Performance visualizations
   - Real-time filtering and analysis

## Feature Pipeline

### Stage 1: Data Loading (`src/feature_pipeline/load.py`)

Implements **time-based data splitting** to prevent data leakage:

- **Train**: Data before 2020-01-01
- **Test**: Data from 2020-01-01 to 2022-01-01
- **Holdout**: Data after 2022-01-01

Outputs to `data/raw/` for the next stage.

### Stage 2: Preprocessing (`src/feature_pipeline/preprocess.py`)

Cleans and normalizes raw housing data:

- **City Name Normalization**: Maps non-standard metro names to canonical forms using `CITY_MAPPING`
- **Geographic Enrichment**: Merges with `raw_usmetros.csv` to add latitude/longitude coordinates
- **Duplicate Removal**: Removes exact duplicates (excluding date/year columns)
- **Outlier Removal**: Filters extreme values (e.g., median_list_price > $19M)
- **Quality Checks**: Validates data integrity

Outputs to `data/processed/` with `cleaning_` prefix.

### Stage 3: Feature Engineering (`src/feature_pipeline/feature_engineering.py`)

Transforms raw features into model-ready representations:

#### Feature Extraction
- **Temporal Features**: Extracts year, month, quarter from date column
- **Location Features**: Utilizes lat/lng from preprocessed data

#### Feature Encoding
- **Frequency Encoding** (Zipcode): Encodes categorical variable by frequency of occurrence
- **Target Encoding** (City): Uses category_encoders.TargetEncoder with median_list_price as target
  - Fitted on training data only to prevent leakage
  - Applied consistently to test/holdout splits

#### Model Persistence
- Fitted encoders saved as `.pkl` files in `models/` directory
- Ensures consistent preprocessing during inference

Outputs to `data/processed/` with `feature_engineered_` prefix.

## Training Pipeline

### Model Training (`src/training_pipeline/train.py`)

Trains LightGBM regressors with comprehensive validation:

```python
from src.training_pipeline.train import train_model

# Train with default parameters
model, metrics = train_model()
# Returns: (LGBMRegressor, {"mae": 0.xx, "rmse": 0.xx, "r2": 0.xx})

# Train with custom parameters
custom_params = {"learning_rate": 0.1, "num_leaves": 50}
model, metrics = train_model(model_params=custom_params)

# Quick iteration with data sampling
model, metrics = train_model(sample_fraction=0.1)
```

**Features**:
- Trains on feature-engineered train split
- Validates on feature-engineered test split
- Supports custom hyperparameters
- Optional data sampling for fast iteration
- Saves trained model to `models/lightgbm_model.pkl`

### Model Evaluation (`src/training_pipeline/test.py`)

Evaluates trained models on held-out test data:

```python
from src.training_pipeline.test import evaluate_model

metrics = evaluate_model()
# Returns: {"mae": 0.xx, "rmse": 0.xx, "r2": 0.xx}
```

**Features**:
- Loads trained model from disk
- Evaluates against test split
- Computes standard regression metrics
- Supports optional data sampling for validation

### Hyperparameter Tuning (`src/training_pipeline/tune.py`)

Optimizes model hyperparameters using Optuna:

```python
uv run python src/training_pipeline/tune.py
```

**Optimization Features**:
- Searches over: num_leaves, learning_rate, lambda_l1, lambda_l2, etc.
- Uses Optuna for efficient hyperparameter search
- Integrates with MLFlow for experiment tracking
- Logs all runs for later comparison and analysis

### Experiment Tracking (MLFlow)

All experiments are logged to MLFlow for tracking and comparison:

```bash
# View MLFlow UI
mlflow ui

# Access at http://localhost:5000
```

Tracks:
- Training parameters
- Evaluation metrics (MAE, RMSE, R²)
- Model artifacts
- Hyperparameter search history

## Containerization & Deployment

### Docker Architecture

The project uses **multi-stage Docker builds** for optimal image size and security:

#### Backend (FastAPI)
- **Stage 1**: Builder - installs dependencies with `uv` package manager
- **Stage 2**: Runtime - minimal `python:3.10-slim` image with only necessary packages
- **Size**: ~400MB
- **Security**: Runs as non-root user

#### Frontend (Streamlit)
- **Stage 1**: Builder - creates virtual environment
- **Stage 2**: Runtime - minimal runtime image
- **Size**: ~500MB
- **Security**: Non-root execution

#### Benefits
- Small image sizes (vs 1GB+ with single-stage builds)
- No build tools in production images
- Docker layer caching for faster rebuilds
- Enhanced security with minimal attack surface

### Local Development with Docker Compose

```bash
# Start all services
docker-compose up -d

# Services available at:
# Frontend: http://localhost:8501
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Production Deployment on AWS

For production deployments with AWS ECS, refer to [DOCKER.md](DOCKER.md) for:
- Amazon ECR (Elastic Container Registry) setup
- AWS ECS (Elastic Container Service) configuration
- Application Load Balancer setup
- Environment variable configuration
- Health checks and monitoring
- Auto-scaling strategies

## Quick Start

### Prerequisites

- **Python**: 3.10+
- **uv**: Package manager ([Install uv](https://docs.astral.sh/uv/))
- **Docker** (optional): For containerized deployment
- **AWS Credentials** (optional): For S3 integration

### Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd Housing-Pricing-End-to-End-ML

# 2. Install dependencies
uv sync

# 3. Create environment file
cat > .env << EOF
# AWS S3 Configuration (optional)
S3_BUCKET=housing-pricing-regression-data
AWS_REGION=us-east-2
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key

# API Configuration
API_URL=http://127.0.0.1:8000/predict

# Optional MLFlow Configuration
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
EOF
```

## Running the Pipelines

### Feature Pipeline

Process raw data through the complete feature engineering pipeline:

```bash
# Step 1: Load and split raw data by time
uv run python src/feature_pipeline/load.py

# Step 2: Preprocess data splits
uv run python src/feature_pipeline/preprocess.py

# Step 3: Feature engineering and encoding
uv run python -c "from src.feature_pipeline.feature_engineering import run_feature_engineering_pipeline; run_feature_engineering_pipeline()"
```

### Training Pipeline

Train and evaluate the LightGBM model:

```bash
# Step 1: Train model
uv run python src/training_pipeline/train.py

# Step 2: Evaluate on test set
uv run python src/training_pipeline/test.py

# Step 3 (Optional): Hyperparameter tuning with Optuna
uv run python src/training_pipeline/tune.py

# View MLFlow experiments
mlflow ui
```

### Backend & Frontend

Run the API server and interactive UI:

```bash
# Terminal 1: Start FastAPI backend
uv run python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000

# Terminal 2: Start Streamlit frontend
uv run streamlit run streamlit_app.py
```

Access:
- **Frontend**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### Make Predictions

```bash
# Health check
curl http://127.0.0.1:8000/health

# Expected response
{
  "status": "Healthy",
  "model_path": "models/lightgbm_model.pkl",
  "n_features_expected": 39
}

# Batch prediction
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '[{
    "date": "2023-01-01",
    "median_sale_price": 300000,
    "median_list_price": 320000,
    ...other_features...
  }]'
```

## Testing

The project includes comprehensive unit and integration tests with **96% code coverage**.

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/test_load.py -v

# Run unit tests only
uv run pytest -m unit

# Run integration tests only
uv run pytest -m integration

# View coverage report
open htmlcov/index.html  # macOS
```

### Test Structure

- `tests/conftest.py`: Shared fixtures
- `tests/test_load.py`: Data loading tests
- `tests/test_preprocess.py`: Preprocessing tests
- `tests/test_feature_engineering.py`: Feature engineering tests
- `tests/test_training_pipeline.py`: Training/evaluation tests

### Test Markers

- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests

## Project Structure

```
housing-pricing-end-to-end-ml/
├── data/
│   ├── raw/                           # Initial data splits
│   │   ├── train_data.csv
│   │   ├── test_data.csv
│   │   ├── holdout_data.csv
│   │   └── raw_usmetros.csv           # Geographic reference data
│   └── processed/                     # Cleaned and engineered data
│       ├── cleaning_{train,test,holdout}_data.csv
│       └── feature_engineered_{train,test,holdout}_data.csv
├── src/
│   ├── api/
│   │   └── main.py                    # FastAPI backend
│   ├── feature_pipeline/
│   │   ├── load.py                    # Time-based data splitting
│   │   ├── preprocess.py              # Data cleaning and normalization
│   │   └── feature_engineering.py     # Feature extraction and encoding
│   ├── training_pipeline/
│   │   ├── train.py                   # Model training
│   │   ├── test.py                    # Model evaluation
│   │   └── tune.py                    # Hyperparameter tuning
│   └── inference_pipeline/
│       └── inference.py               # Batch predictions
├── models/                            # Saved encoders and models (.pkl)
│   ├── lightgbm_model.pkl
│   ├── frequency_encoder.pkl
│   └── target_encoder.pkl
├── notebooks/                         # Jupyter notebooks for exploration
├── tests/                             # Comprehensive test suite
│   ├── conftest.py
│   ├── test_load.py
│   ├── test_preprocess.py
│   ├── test_feature_engineering.py
│   └── test_training_pipeline.py
├── streamlit_app.py                   # Interactive UI
├── Dockerfile.backend                 # Backend containerization
├── Dockerfile.frontend                # Frontend containerization
├── docker-compose.yml                 # Multi-container orchestration
├── pyproject.toml                     # Dependency management
├── pytest.ini                         # pytest configuration
├── AGENTS.md                          # Agent/workflow documentation
├── DOCKER.md                          # Docker deployment guide
├── README.md                          # This file
└── Makefile                           # Common commands
```

## Key Technologies & Dependencies

### Data Processing & ML
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning utilities and metrics
- **LightGBM**: Primary gradient boosting model
- **XGBoost**: Alternative model (for testing/comparison)

### Feature Engineering & Encoding
- **category-encoders**: Target encoding, frequency encoding

### Experiment Tracking & Optimization
- **MLFlow**: Experiment tracking and model registry
- **Optuna**: Hyperparameter optimization framework

### API & Web
- **FastAPI**: High-performance REST API framework
- **Streamlit**: Interactive data science web UI

### Data Quality & Validation
- **great-expectations**: Data quality testing
- **evidently**: ML model and data monitoring

### Cloud & Storage
- **boto3**: AWS S3 integration

### Testing
- **pytest**: Testing framework
- **pytest-cov**: Coverage reporting

### Visualization
- **matplotlib**: Static plots
- **seaborn**: Statistical visualization
- **plotly**: Interactive visualizations

## Key Concepts

### Time-Based Data Splitting

Prevents data leakage by splitting chronologically:
- **Training**: Historical data up to 2020-01-01
- **Testing**: Data from early 2020 to early 2022
- **Holdout**: Recent data after 2022-01-01

### Feature Engineering Strategy

1. **Temporal Features**: Extract temporal patterns (year, month, quarter)
2. **Location Features**: Use geographic coordinates for spatial patterns
3. **Categorical Encoding**:
   - Frequency encoding for popularity indicators
   - Target encoding for city-level price patterns

### Model Persistence

- Encoders fit on training data only
- Saved to disk for consistent inference
- Prevents leakage during preprocessing

### Inference Pipeline

The complete preprocessing pipeline is replicated for inference:
```
Raw Input → Preprocessing → Feature Engineering → Prediction
```

## Troubleshooting

### "No module named src" when running scripts
```bash
# Set PYTHONPATH to project root
PYTHONPATH=. uv run python src/training_pipeline/train.py
```

### Model not found
- Check `models/lightgbm_model.pkl` exists
- Or train: `uv run python src/training_pipeline/train.py`

### API port already in use
```bash
# Use different port
uv run python -m uvicorn src.api.main:app --port 8001
```

### S3 credentials errors
- Ensure AWS credentials in `.env`
- Or run with local model/data cached locally

## Contributing

1. Write tests for new features
2. Maintain 96%+ code coverage
3. Run full test suite before commits: `uv run pytest --cov=src`
4. Follow the existing project structure
5. Use meaningful commit messages

## License

Project repository: [Housing-Pricing-End-to-End-ML](https://github.com/your-repo-url)

For detailed Docker deployment instructions, see [DOCKER.md](DOCKER.md).
