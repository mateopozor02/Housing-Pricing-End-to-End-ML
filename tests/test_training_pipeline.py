"""
Unit and integration tests for the training pipeline (model training and evaluation).
Tests both train.py and test.py modules.
"""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from joblib import dump
import tempfile
import lightgbm as lgb

from src.training_pipeline.test import (
    _maybe_sample as evaluate_maybe_sample,
    evaluate_model,
    DEFAULT_TEST_PATH,
    DEFAULT_MODEL_PATH,
)
from src.training_pipeline.train import (
    _maybe_sample as train_maybe_sample,
    train_model,
    DEFAULT_TRAIN_PATH,
)


# ============================================================================
# Tests for _maybe_sample function (used in both train.py and test.py)
# ============================================================================


@pytest.mark.unit
class TestMaybeSample:
    """Test suite for _maybe_sample function used in training pipeline."""

    def test_returns_full_dataframe_when_none(self):
        """Test that full DataFrame is returned when sample_fraction is None."""
        # Arrange
        df = pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": [10, 20, 30, 40, 50]})

        # Act
        result = train_maybe_sample(df, sample_fraction=None, random_state=42)

        # Assert
        assert len(result) == len(df)
        pd.testing.assert_frame_equal(result, df)

    def test_returns_full_dataframe_when_zero(self):
        """Test that full DataFrame is returned when sample_fraction is 0."""
        # Arrange
        df = pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": [10, 20, 30, 40, 50]})

        # Act
        result = train_maybe_sample(df, sample_fraction=0.0, random_state=42)

        # Assert
        assert len(result) == len(df)

    def test_returns_full_dataframe_when_one(self):
        """Test that full DataFrame is returned when sample_fraction is 1."""
        # Arrange
        df = pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": [10, 20, 30, 40, 50]})

        # Act
        result = train_maybe_sample(df, sample_fraction=1.0, random_state=42)

        # Assert
        assert len(result) == len(df)

    def test_returns_half_sample(self):
        """Test that correct sample size is returned for 0.5."""
        # Arrange
        df = pd.DataFrame({"col1": range(100), "col2": range(100, 200)})

        # Act
        result = train_maybe_sample(df, sample_fraction=0.5, random_state=42)

        # Assert
        assert len(result) == 50

    def test_returns_quarter_sample(self):
        """Test that correct sample size is returned for 0.25."""
        # Arrange
        df = pd.DataFrame({"col1": range(100), "col2": range(100, 200)})

        # Act
        result = train_maybe_sample(df, sample_fraction=0.25, random_state=42)

        # Assert
        assert len(result) == 25

    def test_sample_is_reproducible(self):
        """Test that sampling with same random_state produces same result."""
        # Arrange
        df = pd.DataFrame({"col1": range(100), "col2": range(100, 200)})

        # Act
        result1 = train_maybe_sample(df, sample_fraction=0.5, random_state=42)
        result2 = train_maybe_sample(df, sample_fraction=0.5, random_state=42)

        # Assert
        pd.testing.assert_frame_equal(result1, result2)

    def test_sample_index_reset(self):
        """Test that index is reset after sampling."""
        # Arrange
        df = pd.DataFrame({"col1": range(100), "col2": range(100, 200)})

        # Act
        result = train_maybe_sample(df, sample_fraction=0.5, random_state=42)

        # Assert
        assert list(result.index) == list(range(len(result)))

    def test_preserves_dataframe_columns(self):
        """Test that all columns are preserved after sampling."""
        # Arrange
        df = pd.DataFrame(
            {"col1": range(10), "col2": range(10, 20), "col3": range(20, 30)}
        )

        # Act
        result = train_maybe_sample(df, sample_fraction=0.5, random_state=42)

        # Assert
        assert set(result.columns) == set(df.columns)

    def test_handles_negative_fraction(self):
        """Test that negative fraction returns full DataFrame."""
        # Arrange
        df = pd.DataFrame({"col1": [1, 2, 3, 4, 5]})

        # Act
        result = train_maybe_sample(df, sample_fraction=-0.5, random_state=42)

        # Assert
        assert len(result) == len(df)

    def test_handles_fraction_greater_than_one(self):
        """Test that fraction > 1 returns full DataFrame."""
        # Arrange
        df = pd.DataFrame({"col1": [1, 2, 3, 4, 5]})

        # Act
        result = train_maybe_sample(df, sample_fraction=1.5, random_state=42)

        # Assert
        assert len(result) == len(df)


# ============================================================================
# Tests for train_model function
# ============================================================================


@pytest.mark.unit
class TestTrainModel:
    """Test suite for train_model function."""

    @pytest.fixture
    def sample_train_data_large(self):
        """Create sample training data for model training."""
        np.random.seed(42)
        n_samples = 200
        return pd.DataFrame(
            {
                "price": np.random.uniform(100000, 1000000, n_samples),
                "feature1": np.random.uniform(0, 100, n_samples),
                "feature2": np.random.uniform(0, 100, n_samples),
                "feature3": np.random.uniform(0, 100, n_samples),
                "lat": np.random.uniform(25, 50, n_samples),
                "lng": np.random.uniform(-125, -65, n_samples),
            }
        )

    @pytest.fixture
    def sample_test_data_for_train(self):
        """Create sample test data for model validation."""
        np.random.seed(42)
        n_samples = 50
        return pd.DataFrame(
            {
                "price": np.random.uniform(100000, 1000000, n_samples),
                "feature1": np.random.uniform(0, 100, n_samples),
                "feature2": np.random.uniform(0, 100, n_samples),
                "feature3": np.random.uniform(0, 100, n_samples),
                "lat": np.random.uniform(25, 50, n_samples),
                "lng": np.random.uniform(-125, -65, n_samples),
            }
        )

    def test_returns_trained_model_and_metrics(
        self, sample_train_data_large, sample_test_data_for_train, tmp_path
    ):
        """Test that train_model returns a model and metrics dictionary."""
        # Arrange
        train_path = tmp_path / "train_data.csv"
        test_path = tmp_path / "test_data.csv"
        model_output = tmp_path / "model.pkl"

        sample_train_data_large.to_csv(train_path, index=False)
        sample_test_data_for_train.to_csv(test_path, index=False)

        # Act
        model, metrics = train_model(
            train_path=train_path, test_path=test_path, model_output=model_output
        )

        # Assert
        assert model is not None
        assert isinstance(metrics, dict)
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics

    def test_metrics_are_floats(
        self, sample_train_data_large, sample_test_data_for_train, tmp_path
    ):
        """Test that all returned metrics are float values."""
        # Arrange
        train_path = tmp_path / "train_data.csv"
        test_path = tmp_path / "test_data.csv"
        model_output = tmp_path / "model.pkl"

        sample_train_data_large.to_csv(train_path, index=False)
        sample_test_data_for_train.to_csv(test_path, index=False)

        # Act
        model, metrics = train_model(
            train_path=train_path, test_path=test_path, model_output=model_output
        )

        # Assert
        assert isinstance(metrics["mae"], (float, np.floating))
        assert isinstance(metrics["rmse"], (float, np.floating))
        assert isinstance(metrics["r2"], (float, np.floating))

    def test_model_saved_to_disk(
        self, sample_train_data_large, sample_test_data_for_train, tmp_path
    ):
        """Test that trained model is saved to specified output path."""
        # Arrange
        train_path = tmp_path / "train_data.csv"
        test_path = tmp_path / "test_data.csv"
        model_output = tmp_path / "model.pkl"

        sample_train_data_large.to_csv(train_path, index=False)
        sample_test_data_for_train.to_csv(test_path, index=False)

        # Act
        model, metrics = train_model(
            train_path=train_path, test_path=test_path, model_output=model_output
        )

        # Assert
        assert model_output.exists()

    def test_uses_sample_fraction(
        self, sample_train_data_large, sample_test_data_for_train, tmp_path
    ):
        """Test that sample_fraction parameter is respected."""
        # Arrange
        train_path = tmp_path / "train_data.csv"
        test_path = tmp_path / "test_data.csv"
        model_output = tmp_path / "model.pkl"

        sample_train_data_large.to_csv(train_path, index=False)
        sample_test_data_for_train.to_csv(test_path, index=False)

        # Act - should not raise an error
        model, metrics = train_model(
            train_path=train_path,
            test_path=test_path,
            model_output=model_output,
            sample_fraction=0.5,
        )

        # Assert
        assert model is not None
        assert isinstance(metrics, dict)

    def test_accepts_custom_model_params(
        self, sample_train_data_large, sample_test_data_for_train, tmp_path
    ):
        """Test that custom model parameters are accepted."""
        # Arrange
        train_path = tmp_path / "train_data.csv"
        test_path = tmp_path / "test_data.csv"
        model_output = tmp_path / "model.pkl"
        custom_params = {"num_leaves": 31, "learning_rate": 0.1, "n_estimators": 100}

        sample_train_data_large.to_csv(train_path, index=False)
        sample_test_data_for_train.to_csv(test_path, index=False)

        # Act
        model, metrics = train_model(
            train_path=train_path,
            test_path=test_path,
            model_output=model_output,
            model_params=custom_params,
        )

        # Assert
        assert model is not None
        assert isinstance(metrics, dict)

    def test_returns_positive_metrics(
        self, sample_train_data_large, sample_test_data_for_train, tmp_path
    ):
        """Test that MAE and RMSE are non-negative."""
        # Arrange
        train_path = tmp_path / "train_data.csv"
        test_path = tmp_path / "test_data.csv"
        model_output = tmp_path / "model.pkl"

        sample_train_data_large.to_csv(train_path, index=False)
        sample_test_data_for_train.to_csv(test_path, index=False)

        # Act
        model, metrics = train_model(
            train_path=train_path, test_path=test_path, model_output=model_output
        )

        # Assert
        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= 0


# ============================================================================
# Tests for evaluate_model function
# ============================================================================


@pytest.mark.unit
class TestEvaluateModel:
    """Test suite for evaluate_model function."""

    @pytest.fixture
    def sample_test_data_eval(self):
        """Create sample test data for evaluation."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "price": np.random.uniform(100000, 1000000, 100),
                "feature1": np.random.uniform(0, 100, 100),
                "feature2": np.random.uniform(0, 100, 100),
                "feature3": np.random.uniform(0, 100, 100),
                "lat": np.random.uniform(25, 50, 100),
                "lng": np.random.uniform(-125, -65, 100),
            }
        )

    def test_returns_metrics_dict(self, sample_test_data_eval, tmp_path):
        """Test that evaluate_model returns a dictionary with correct keys."""
        # Arrange
        test_path = tmp_path / "test_data.csv"
        model_path = tmp_path / "model.pkl"
        sample_test_data_eval.to_csv(test_path, index=False)

        mock_model = Mock()
        mock_model.best_iteration = 100
        mock_model.predict = Mock(return_value=np.random.uniform(100000, 1000000, 100))

        # Act
        with patch("src.training_pipeline.test.load", return_value=mock_model):
            result = evaluate_model(model_path=model_path, test_path=test_path)

        # Assert
        assert isinstance(result, dict)
        assert "mae" in result
        assert "rmse" in result
        assert "r2" in result

    def test_metrics_are_floats(self, sample_test_data_eval, tmp_path):
        """Test that all metrics are float values."""
        # Arrange
        test_path = tmp_path / "test_data.csv"
        model_path = tmp_path / "model.pkl"
        sample_test_data_eval.to_csv(test_path, index=False)

        mock_model = Mock()
        mock_model.best_iteration = 100
        mock_model.predict = Mock(return_value=np.random.uniform(100000, 1000000, 100))

        # Act
        with patch("src.training_pipeline.test.load", return_value=mock_model):
            result = evaluate_model(model_path=model_path, test_path=test_path)

        # Assert
        assert isinstance(result["mae"], (float, np.floating))
        assert isinstance(result["rmse"], (float, np.floating))
        assert isinstance(result["r2"], (float, np.floating))

    def test_mae_is_non_negative(self, sample_test_data_eval, tmp_path):
        """Test that MAE is non-negative."""
        # Arrange
        test_path = tmp_path / "test_data.csv"
        model_path = tmp_path / "model.pkl"
        sample_test_data_eval.to_csv(test_path, index=False)

        mock_model = Mock()
        mock_model.best_iteration = 100
        mock_model.predict = Mock(return_value=np.random.uniform(100000, 1000000, 100))

        # Act
        with patch("src.training_pipeline.test.load", return_value=mock_model):
            result = evaluate_model(model_path=model_path, test_path=test_path)

        # Assert
        assert result["mae"] >= 0

    def test_rmse_is_non_negative(self, sample_test_data_eval, tmp_path):
        """Test that RMSE is non-negative."""
        # Arrange
        test_path = tmp_path / "test_data.csv"
        model_path = tmp_path / "model.pkl"
        sample_test_data_eval.to_csv(test_path, index=False)

        mock_model = Mock()
        mock_model.best_iteration = 100
        mock_model.predict = Mock(return_value=np.random.uniform(100000, 1000000, 100))

        # Act
        with patch("src.training_pipeline.test.load", return_value=mock_model):
            result = evaluate_model(model_path=model_path, test_path=test_path)

        # Assert
        assert result["rmse"] >= 0

    def test_with_sample_fraction(self, sample_test_data_eval, tmp_path):
        """Test evaluation with sample_fraction parameter."""
        # Arrange
        test_path = tmp_path / "test_data.csv"
        model_path = tmp_path / "model.pkl"
        sample_test_data_eval.to_csv(test_path, index=False)

        mock_model = Mock()
        mock_model.best_iteration = 100
        mock_model.predict = Mock(return_value=np.random.uniform(100000, 1000000, 50))

        # Act
        with patch("src.training_pipeline.test.load", return_value=mock_model):
            result = evaluate_model(
                model_path=model_path, test_path=test_path, sample_fraction=0.5
            )

        # Assert
        assert isinstance(result, dict)
        assert "mae" in result

    def test_uses_best_iteration(self, sample_test_data_eval, tmp_path):
        """Test that model uses best_iteration parameter."""
        # Arrange
        test_path = tmp_path / "test_data.csv"
        model_path = tmp_path / "model.pkl"
        sample_test_data_eval.to_csv(test_path, index=False)

        mock_model = Mock()
        mock_model.best_iteration = 150
        mock_model.predict = Mock(return_value=np.random.uniform(100000, 1000000, 100))

        # Act
        with patch("src.training_pipeline.test.load", return_value=mock_model):
            evaluate_model(model_path=model_path, test_path=test_path)

        # Assert
        mock_model.predict.assert_called()
        call_kwargs = mock_model.predict.call_args[1]
        assert call_kwargs.get("num_iteration") == 150

    def test_drops_price_column_for_features(self, sample_test_data_eval, tmp_path):
        """Test that price column is dropped when creating feature matrix."""
        # Arrange
        test_path = tmp_path / "test_data.csv"
        model_path = tmp_path / "model.pkl"
        sample_test_data_eval.to_csv(test_path, index=False)

        mock_model = Mock()
        mock_model.best_iteration = 100
        mock_model.predict = Mock(return_value=np.random.uniform(100000, 1000000, 100))

        # Act
        with patch("src.training_pipeline.test.load", return_value=mock_model):
            evaluate_model(model_path=model_path, test_path=test_path)

        # Assert
        call_args = mock_model.predict.call_args[0][0]
        assert "price" not in call_args.columns


# ============================================================================
# Integration tests for training pipeline
# ============================================================================


@pytest.mark.integration
class TestTrainingPipelineIntegration:
    """Integration tests for the complete training pipeline."""

    def test_full_training_pipeline(self, tmp_path):
        """Test complete training pipeline from data to saved model."""
        # Arrange
        np.random.seed(42)
        n_train = 200
        n_test = 50

        train_df = pd.DataFrame(
            {
                "price": np.random.uniform(100000, 1000000, n_train),
                "feature1": np.random.uniform(0, 100, n_train),
                "feature2": np.random.uniform(0, 100, n_train),
                "feature3": np.random.uniform(0, 100, n_train),
                "lat": np.random.uniform(25, 50, n_train),
                "lng": np.random.uniform(-125, -65, n_train),
            }
        )

        test_df = pd.DataFrame(
            {
                "price": np.random.uniform(100000, 1000000, n_test),
                "feature1": np.random.uniform(0, 100, n_test),
                "feature2": np.random.uniform(0, 100, n_test),
                "feature3": np.random.uniform(0, 100, n_test),
                "lat": np.random.uniform(25, 50, n_test),
                "lng": np.random.uniform(-125, -65, n_test),
            }
        )

        train_path = tmp_path / "train_data.csv"
        test_path = tmp_path / "test_data.csv"
        model_output = tmp_path / "model.pkl"

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        # Act - Train model
        model, train_metrics = train_model(
            train_path=train_path, test_path=test_path, model_output=model_output
        )

        # Act - Evaluate model
        eval_metrics = evaluate_model(model_path=model_output, test_path=test_path)

        # Assert
        assert model is not None
        assert model_output.exists()
        assert "mae" in train_metrics
        assert "mae" in eval_metrics
        assert eval_metrics["mae"] >= 0
        assert eval_metrics["rmse"] >= 0

    def test_model_can_be_reloaded_and_evaluated(self, tmp_path):
        """Test that trained model can be reloaded and evaluated."""
        # Arrange
        np.random.seed(42)
        n_train = 200
        n_test = 50

        train_df = pd.DataFrame(
            {
                "price": np.random.uniform(100000, 1000000, n_train),
                "feature1": np.random.uniform(0, 100, n_train),
                "feature2": np.random.uniform(0, 100, n_train),
                "lat": np.random.uniform(25, 50, n_train),
                "lng": np.random.uniform(-125, -65, n_train),
            }
        )

        test_df = pd.DataFrame(
            {
                "price": np.random.uniform(100000, 1000000, n_test),
                "feature1": np.random.uniform(0, 100, n_test),
                "feature2": np.random.uniform(0, 100, n_test),
                "lat": np.random.uniform(25, 50, n_test),
                "lng": np.random.uniform(-125, -65, n_test),
            }
        )

        train_path = tmp_path / "train_data.csv"
        test_path = tmp_path / "test_data.csv"
        model_output = tmp_path / "model.pkl"

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        # Act - Train and save model
        model, _ = train_model(
            train_path=train_path, test_path=test_path, model_output=model_output
        )

        # Act - Reload model by specifying the same path
        metrics = evaluate_model(model_path=model_output, test_path=test_path)

        # Assert
        assert isinstance(metrics, dict)
        assert "mae" in metrics

    def test_training_with_sample_fraction(self, tmp_path):
        """Test training and evaluation with sample fraction for quick iteration."""
        # Arrange
        np.random.seed(42)
        n_train = 200
        n_test = 50

        train_df = pd.DataFrame(
            {
                "price": np.random.uniform(100000, 1000000, n_train),
                "feature1": np.random.uniform(0, 100, n_train),
                "feature2": np.random.uniform(0, 100, n_train),
                "lat": np.random.uniform(25, 50, n_train),
                "lng": np.random.uniform(-125, -65, n_train),
            }
        )

        test_df = pd.DataFrame(
            {
                "price": np.random.uniform(100000, 1000000, n_test),
                "feature1": np.random.uniform(0, 100, n_test),
                "feature2": np.random.uniform(0, 100, n_test),
                "lat": np.random.uniform(25, 50, n_test),
                "lng": np.random.uniform(-125, -65, n_test),
            }
        )

        train_path = tmp_path / "train_data.csv"
        test_path = tmp_path / "test_data.csv"
        model_output = tmp_path / "model.pkl"

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        # Act - Train with sample fraction
        model, train_metrics = train_model(
            train_path=train_path,
            test_path=test_path,
            model_output=model_output,
            sample_fraction=0.5,
        )

        # Act - Evaluate with sample fraction
        eval_metrics = evaluate_model(
            model_path=model_output, test_path=test_path, sample_fraction=0.5
        )

        # Assert
        assert model is not None
        assert "mae" in train_metrics
        assert "mae" in eval_metrics
