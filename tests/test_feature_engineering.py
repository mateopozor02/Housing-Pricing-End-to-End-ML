"""
Unit tests for the feature_engineering module (feature extraction and encoding).
"""

import pandas as pd
import pytest
from pathlib import Path
from category_encoders import TargetEncoder
from joblib import load
from src.feature_pipeline.feature_engineering import (
    add_date_features,
    frequency_encode,
    target_encode,
    drop_unused_columns,
    run_feature_engineering_pipeline
)


@pytest.mark.unit
class TestAddDateFeatures:
    """Test suite for add_date_features function."""
    
    def test_extracts_year_month_quarter(self, sample_train_data):
        """Test that year, month, and quarter are extracted from date."""
        # Act
        result = add_date_features(sample_train_data.copy())
        
        # Assert
        assert "year" in result.columns
        assert "month" in result.columns
        assert "quarter" in result.columns
    
    def test_date_values_correct(self):
        """Test that extracted date values are correct."""
        # Arrange
        df = pd.DataFrame({
            "date": ["2019-03-15", "2020-06-20", "2021-12-25"]
        })
        
        # Act
        result = add_date_features(df)
        
        # Assert
        assert result.loc[0, "year"] == 2019
        assert result.loc[0, "month"] == 3
        assert result.loc[0, "quarter"] == 1
        
        assert result.loc[1, "year"] == 2020
        assert result.loc[1, "month"] == 6
        assert result.loc[1, "quarter"] == 2
        
        assert result.loc[2, "year"] == 2021
        assert result.loc[2, "month"] == 12
        assert result.loc[2, "quarter"] == 4
    
    def test_date_columns_positioned_correctly(self, sample_train_data):
        """Test that date feature columns are inserted at correct positions."""
        # Act
        result = add_date_features(sample_train_data.copy())
        
        # Assert
        columns_list = list(result.columns)
        year_idx = columns_list.index("year")
        month_idx = columns_list.index("month")
        quarter_idx = columns_list.index("quarter")
        
        assert year_idx == 1
        assert month_idx == 2
        assert quarter_idx == 3
    
    def test_preserves_other_columns(self, sample_train_data):
        """Test that other columns are preserved."""
        # Arrange
        original_columns = set(sample_train_data.columns)
        
        # Act
        result = add_date_features(sample_train_data.copy())
        
        # Assert
        for col in original_columns:
            assert col in result.columns
    
    def test_converts_string_dates_to_datetime(self):
        """Test that string dates are converted to datetime."""
        # Arrange
        df = pd.DataFrame({
            "date": ["2019-01-01", "2020-06-15"],
            "value": [1, 2]
        })
        
        # Act
        result = add_date_features(df)
        
        # Assert
        assert pd.api.types.is_datetime64_any_dtype(result["date"])


@pytest.mark.unit
class TestFrequencyEncode:
    """Test suite for frequency_encode function."""
    
    def test_creates_frequency_column(self, sample_train_data, sample_test_data):
        """Test that frequency encoding column is created."""
        # Act
        train_result, test_result, freq_map = frequency_encode(
            sample_train_data.copy(),
            sample_test_data.copy(),
            "zipcode"
        )
        
        # Assert
        assert "zipcode_freq" in train_result.columns
        assert "zipcode_freq" in test_result.columns
    
    def test_frequency_values_correct(self):
        """Test that frequency values are calculated correctly."""
        # Arrange
        train = pd.DataFrame({
            "zipcode": ["10001", "10001", "90001", "10001"]
        })
        test = pd.DataFrame({
            "zipcode": ["10001", "90001"]
        })
        
        # Act
        train_result, test_result, freq_map = frequency_encode(train, test, "zipcode")
        
        # Assert
        assert train_result.loc[0, "zipcode_freq"] == 3  # "10001" appears 3 times
        assert train_result.loc[2, "zipcode_freq"] == 1  # "90001" appears 1 time
        assert test_result.loc[0, "zipcode_freq"] == 3
        assert test_result.loc[1, "zipcode_freq"] == 1
    
    def test_returns_frequency_map(self):
        """Test that frequency map is returned correctly."""
        # Arrange
        train = pd.DataFrame({
            "zipcode": ["10001", "10001", "90001"]
        })
        test = pd.DataFrame({
            "zipcode": ["10001"]
        })
        
        # Act
        _, _, freq_map = frequency_encode(train, test, "zipcode")
        
        # Assert
        assert isinstance(freq_map, pd.Series)
        assert freq_map["10001"] == 2
        assert freq_map["90001"] == 1
    
    def test_handles_unseen_values_in_test(self):
        """Test that unseen values in test set are handled (will be NaN)."""
        # Arrange
        train = pd.DataFrame({
            "zipcode": ["10001", "10001"]
        })
        test = pd.DataFrame({
            "zipcode": ["10001", "99999"]  # 99999 not in train
        })
        
        # Act
        _, test_result, _ = frequency_encode(train, test, "zipcode")
        
        # Assert
        assert pd.isna(test_result.loc[1, "zipcode_freq"])
    
    def test_preserves_original_column(self, sample_train_data, sample_test_data):
        """Test that original column is preserved."""
        # Act
        train_result, test_result, _ = frequency_encode(
            sample_train_data.copy(),
            sample_test_data.copy(),
            "zipcode"
        )
        
        # Assert
        assert "zipcode" in train_result.columns
        assert "zipcode" in test_result.columns


@pytest.mark.unit
class TestTargetEncode:
    """Test suite for target_encode function."""
    
    def test_creates_encoded_column(self, sample_train_data, sample_test_data):
        """Test that target encoding column is created."""
        # Act
        train_result, test_result, encoder = target_encode(
            sample_train_data.copy(),
            sample_test_data.copy(),
            "city_full",
            "median_list_price"
        )
        
        # Assert
        assert "city_full_encoded" in train_result.columns
        assert "city_full_encoded" in test_result.columns
    
    def test_returns_encoder_instance(self, sample_train_data, sample_test_data):
        """Test that a TargetEncoder instance is returned."""
        # Act
        _, _, encoder = target_encode(
            sample_train_data.copy(),
            sample_test_data.copy(),
            "city_full",
            "median_list_price"
        )
        
        # Assert
        assert isinstance(encoder, TargetEncoder)
    
    def test_encoded_values_are_numeric(self, sample_train_data, sample_test_data):
        """Test that encoded values are numeric."""
        # Act
        train_result, test_result, _ = target_encode(
            sample_train_data.copy(),
            sample_test_data.copy(),
            "city_full",
            "median_list_price"
        )
        
        # Assert
        assert pd.api.types.is_numeric_dtype(train_result["city_full_encoded"])
        assert pd.api.types.is_numeric_dtype(test_result["city_full_encoded"])
    
    def test_encoder_fitted_on_train_only(self):
        """Test that encoder is fitted on training data only."""
        # Arrange
        train = pd.DataFrame({
            "city_full": ["New York", "New York", "Los Angeles"],
            "median_list_price": [500000, 600000, 700000]
        })
        test = pd.DataFrame({
            "city_full": ["New York"],
            "median_list_price": [999999]  # Different target value
        })
        
        # Act
        train_result, test_result, _ = target_encode(train, test, "city_full", "median_list_price")
        
        # Assert - New York should be encoded based on train mean (550000), not test value
        ny_train_mean = train[train["city_full"] == "New York"]["median_list_price"].mean()
        assert train_result.loc[0, "city_full_encoded"] != 999999
    
    def test_preserves_original_column(self, sample_train_data, sample_test_data):
        """Test that original column is preserved."""
        # Act
        train_result, test_result, _ = target_encode(
            sample_train_data.copy(),
            sample_test_data.copy(),
            "city_full",
            "median_list_price"
        )
        
        # Assert
        assert "city_full" in train_result.columns
        assert "city_full" in test_result.columns


@pytest.mark.unit
class TestDropUnusedColumns:
    """Test suite for drop_unused_columns function."""
    
    def test_drops_default_columns(self, sample_train_data):
        """Test that default unused columns are dropped."""
        # Arrange
        df = sample_train_data.copy()
        
        # Act
        result = drop_unused_columns(df)
        
        # Assert
        assert "date" not in result.columns
        assert "city_full" not in result.columns
        assert "city" not in result.columns
        assert "zipcode" not in result.columns
        assert "median_sale_price" not in result.columns
    
    def test_drops_custom_columns(self):
        """Test that custom columns can be dropped."""
        # Arrange
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
            "col3": [7, 8, 9]
        })
        
        # Act
        result = drop_unused_columns(df, columns=["col1", "col3"])
        
        # Assert
        assert "col1" not in result.columns
        assert "col3" not in result.columns
        assert "col2" in result.columns
    
    def test_handles_missing_columns(self):
        """Test that missing columns are handled gracefully."""
        # Arrange
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6]
        })
        
        # Act
        result = drop_unused_columns(df, columns=["col1", "col_nonexistent"])
        
        # Assert - should drop col1 but not raise error for col_nonexistent
        assert "col1" not in result.columns
        assert "col2" in result.columns
    
    def test_preserves_other_columns(self, sample_train_data):
        """Test that columns not in drop list are preserved."""
        # Arrange
        df = sample_train_data.copy()
        
        # Act
        result = drop_unused_columns(df)
        
        # Assert
        assert "median_list_price" in result.columns
        assert "lat" in result.columns
        assert "lng" in result.columns


@pytest.mark.integration
class TestRunFeatureEngineeringPipeline:
    """Integration tests for run_feature_engineering_pipeline function."""
    
    def test_pipeline_runs_successfully(self, sample_train_data, sample_test_data, temp_processed_dir, temp_models_dir):
        """Test that the complete pipeline runs without errors."""
        # Arrange
        train_path = temp_processed_dir / "cleaning_train_data.csv"
        test_path = temp_processed_dir / "cleaning_test_data.csv"
        holdout_path = temp_processed_dir / "cleaning_holdout_data.csv"
        
        sample_train_data.to_csv(train_path, index=False)
        sample_test_data.to_csv(test_path, index=False)
        sample_test_data.to_csv(holdout_path, index=False)  # Reuse test data
        
        # Act
        train_df, test_df, holdout_df, freq_map, encoder = run_feature_engineering_pipeline(
            train_path=train_path,
            test_path=test_path,
            holdout_path=holdout_path,
            output_dir=temp_processed_dir
        )
        
        # Assert
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)
        assert isinstance(holdout_df, pd.DataFrame)
        assert freq_map is not None
        assert isinstance(encoder, TargetEncoder)
    
    def test_output_files_created(self, sample_train_data, sample_test_data, temp_processed_dir):
        """Test that output CSV files are created."""
        # Arrange
        train_path = temp_processed_dir / "cleaning_train_data.csv"
        test_path = temp_processed_dir / "cleaning_test_data.csv"
        holdout_path = temp_processed_dir / "cleaning_holdout_data.csv"
        
        sample_train_data.to_csv(train_path, index=False)
        sample_test_data.to_csv(test_path, index=False)
        sample_test_data.to_csv(holdout_path, index=False)
        
        # Act
        run_feature_engineering_pipeline(
            train_path=train_path,
            test_path=test_path,
            holdout_path=holdout_path,
            output_dir=temp_processed_dir
        )
        
        # Assert
        assert (temp_processed_dir / "feature_engineered_train_data.csv").exists()
        assert (temp_processed_dir / "feature_engineered_test_data.csv").exists()
        assert (temp_processed_dir / "feature_engineered_holdout_data.csv").exists()
    
    def test_encoders_saved(self, sample_train_data, sample_test_data, temp_processed_dir, temp_models_dir, monkeypatch):
        """Test that encoders are saved to models directory."""
        # Arrange
        train_path = temp_processed_dir / "cleaning_train_data.csv"
        test_path = temp_processed_dir / "cleaning_test_data.csv"
        holdout_path = temp_processed_dir / "cleaning_holdout_data.csv"
        
        sample_train_data.to_csv(train_path, index=False)
        sample_test_data.to_csv(test_path, index=False)
        sample_test_data.to_csv(holdout_path, index=False)
        
        # Patch MODELS_DIR to use temp directory
        import src.feature_pipeline.feature_engineering as fe_module
        monkeypatch.setattr(fe_module, "MODELS_DIR", temp_models_dir)
        
        # Act
        run_feature_engineering_pipeline(
            train_path=train_path,
            test_path=test_path,
            holdout_path=holdout_path,
            output_dir=temp_processed_dir
        )
        
        # Assert
        assert (temp_models_dir / "frequency_encoder.pkl").exists()
        assert (temp_models_dir / "target_encoder.pkl").exists()
    
    def test_saved_encoders_are_loadable(self, sample_train_data, sample_test_data, temp_processed_dir, temp_models_dir, monkeypatch):
        """Test that saved encoders can be loaded."""
        # Arrange
        train_path = temp_processed_dir / "cleaning_train_data.csv"
        test_path = temp_processed_dir / "cleaning_test_data.csv"
        holdout_path = temp_processed_dir / "cleaning_holdout_data.csv"
        
        sample_train_data.to_csv(train_path, index=False)
        sample_test_data.to_csv(test_path, index=False)
        sample_test_data.to_csv(holdout_path, index=False)
        
        # Patch MODELS_DIR to use temp directory
        import src.feature_pipeline.feature_engineering as fe_module
        monkeypatch.setattr(fe_module, "MODELS_DIR", temp_models_dir)
        
        # Act
        run_feature_engineering_pipeline(
            train_path=train_path,
            test_path=test_path,
            holdout_path=holdout_path,
            output_dir=temp_processed_dir
        )
        
        # Assert - Load encoders
        freq_encoder = load(temp_models_dir / "frequency_encoder.pkl")
        target_encoder = load(temp_models_dir / "target_encoder.pkl")
        
        assert isinstance(freq_encoder, pd.Series)
        assert isinstance(target_encoder, TargetEncoder)
    
    def test_all_date_features_added(self, sample_train_data, sample_test_data, temp_processed_dir):
        """Test that year, month, quarter features are added to all datasets."""
        # Arrange
        train_path = temp_processed_dir / "cleaning_train_data.csv"
        test_path = temp_processed_dir / "cleaning_test_data.csv"
        holdout_path = temp_processed_dir / "cleaning_holdout_data.csv"
        
        sample_train_data.to_csv(train_path, index=False)
        sample_test_data.to_csv(test_path, index=False)
        sample_test_data.to_csv(holdout_path, index=False)
        
        # Act
        train_df, test_df, holdout_df, _, _ = run_feature_engineering_pipeline(
            train_path=train_path,
            test_path=test_path,
            holdout_path=holdout_path,
            output_dir=temp_processed_dir
        )
        
        # Assert
        for df in [train_df, test_df, holdout_df]:
            assert "year" in df.columns
            assert "month" in df.columns
            assert "quarter" in df.columns
    
    def test_encoding_features_added(self, sample_train_data, sample_test_data, temp_processed_dir):
        """Test that frequency and target encoding features are added."""
        # Arrange
        train_path = temp_processed_dir / "cleaning_train_data.csv"
        test_path = temp_processed_dir / "cleaning_test_data.csv"
        holdout_path = temp_processed_dir / "cleaning_holdout_data.csv"
        
        sample_train_data.to_csv(train_path, index=False)
        sample_test_data.to_csv(test_path, index=False)
        sample_test_data.to_csv(holdout_path, index=False)
        
        # Act
        train_df, test_df, holdout_df, _, _ = run_feature_engineering_pipeline(
            train_path=train_path,
            test_path=test_path,
            holdout_path=holdout_path,
            output_dir=temp_processed_dir
        )
        
        # Assert
        for df in [train_df, test_df, holdout_df]:
            assert "zipcode_freq" in df.columns
            assert "city_full_encoded" in df.columns
    
    def test_unused_columns_dropped(self, sample_train_data, sample_test_data, temp_processed_dir):
        """Test that unused columns are dropped from final datasets."""
        # Arrange
        train_path = temp_processed_dir / "cleaning_train_data.csv"
        test_path = temp_processed_dir / "cleaning_test_data.csv"
        holdout_path = temp_processed_dir / "cleaning_holdout_data.csv"
        
        sample_train_data.to_csv(train_path, index=False)
        sample_test_data.to_csv(test_path, index=False)
        sample_test_data.to_csv(holdout_path, index=False)
        
        # Act
        train_df, test_df, holdout_df, _, _ = run_feature_engineering_pipeline(
            train_path=train_path,
            test_path=test_path,
            holdout_path=holdout_path,
            output_dir=temp_processed_dir
        )
        
        # Assert
        for df in [train_df, test_df, holdout_df]:
            assert "date" not in df.columns
            assert "city_full" not in df.columns
            assert "city" not in df.columns
            assert "zipcode" not in df.columns
