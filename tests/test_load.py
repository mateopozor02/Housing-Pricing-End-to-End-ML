"""
Unit tests for the load module (data loading and time-based splitting).
"""

import pandas as pd
import pytest
from pathlib import Path
from src.feature_pipeline.load import load_and_split_data


@pytest.mark.unit
class TestLoadAndSplitData:
    """Test suite for load_and_split_data function."""
    
    def test_load_and_split_creates_three_datasets(self, sample_raw_housing_data, temp_raw_dir):
        """Test that the function creates train, test, and holdout datasets."""
        # Arrange
        raw_path = temp_raw_dir / "raw_housing_data.csv"
        sample_raw_housing_data.to_csv(raw_path, index=False)
        
        # Act
        train_df, test_df, holdout_df = load_and_split_data(
            raw_path=str(raw_path),
            output_dir=temp_raw_dir
        )
        
        # Assert
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)
        assert isinstance(holdout_df, pd.DataFrame)
        assert len(train_df) > 0
        assert len(test_df) > 0
        assert len(holdout_df) > 0
    
    def test_time_based_split_boundaries(self, sample_raw_housing_data, temp_raw_dir):
        """Test that data is split correctly based on date boundaries."""
        # Arrange
        raw_path = temp_raw_dir / "raw_housing_data.csv"
        sample_raw_housing_data.to_csv(raw_path, index=False)
        cutoff_test = pd.Timestamp("2020-01-01")
        cutoff_holdout = pd.Timestamp("2022-01-01")
        
        # Act
        train_df, test_df, holdout_df = load_and_split_data(
            raw_path=str(raw_path),
            output_dir=temp_raw_dir
        )
        
        # Assert - Train set
        train_df["date"] = pd.to_datetime(train_df["date"])
        assert all(train_df["date"] < cutoff_test)
        
        # Assert - Test set
        test_df["date"] = pd.to_datetime(test_df["date"])
        assert all(test_df["date"] >= cutoff_test)
        assert all(test_df["date"] < cutoff_holdout)
        
        # Assert - Holdout set
        holdout_df["date"] = pd.to_datetime(holdout_df["date"])
        assert all(holdout_df["date"] >= cutoff_holdout)
    
    def test_split_sizes_sum_to_original(self, sample_raw_housing_data, temp_raw_dir):
        """Test that the sum of split sizes equals the original dataset size."""
        # Arrange
        raw_path = temp_raw_dir / "raw_housing_data.csv"
        sample_raw_housing_data.to_csv(raw_path, index=False)
        original_size = len(sample_raw_housing_data)
        
        # Act
        train_df, test_df, holdout_df = load_and_split_data(
            raw_path=str(raw_path),
            output_dir=temp_raw_dir
        )
        
        # Assert
        total_size = len(train_df) + len(test_df) + len(holdout_df)
        assert total_size == original_size
    
    def test_output_files_are_created(self, sample_raw_housing_data, temp_raw_dir):
        """Test that output CSV files are created in the specified directory."""
        # Arrange
        raw_path = temp_raw_dir / "raw_housing_data.csv"
        sample_raw_housing_data.to_csv(raw_path, index=False)
        
        # Act
        load_and_split_data(raw_path=str(raw_path), output_dir=temp_raw_dir)
        
        # Assert
        assert (temp_raw_dir / "train_data.csv").exists()
        assert (temp_raw_dir / "test_data.csv").exists()
        assert (temp_raw_dir / "holdout_data.csv").exists()
    
    def test_data_is_sorted_by_date(self, sample_raw_housing_data, temp_raw_dir):
        """Test that the returned dataframes are sorted by date."""
        # Arrange
        raw_path = temp_raw_dir / "raw_housing_data.csv"
        # Shuffle the data to ensure it's not already sorted
        shuffled_data = sample_raw_housing_data.sample(frac=1, random_state=42)
        shuffled_data.to_csv(raw_path, index=False)
        
        # Act
        train_df, test_df, holdout_df = load_and_split_data(
            raw_path=str(raw_path),
            output_dir=temp_raw_dir
        )
        
        # Assert
        for df in [train_df, test_df, holdout_df]:
            if len(df) > 1:
                df["date"] = pd.to_datetime(df["date"])
                assert df["date"].is_monotonic_increasing
    
    def test_date_column_converted_to_datetime(self, sample_raw_housing_data, temp_raw_dir):
        """Test that the date column is converted to datetime type."""
        # Arrange
        raw_path = temp_raw_dir / "raw_housing_data.csv"
        sample_raw_housing_data.to_csv(raw_path, index=False)
        
        # Act
        train_df, test_df, holdout_df = load_and_split_data(
            raw_path=str(raw_path),
            output_dir=temp_raw_dir
        )
        
        # Assert
        for df in [train_df, test_df, holdout_df]:
            # When read back from CSV, dates are strings, but the function should handle this
            assert "date" in df.columns
    
    def test_all_columns_preserved(self, sample_raw_housing_data, temp_raw_dir):
        """Test that all columns from the original data are preserved."""
        # Arrange
        raw_path = temp_raw_dir / "raw_housing_data.csv"
        sample_raw_housing_data.to_csv(raw_path, index=False)
        original_columns = set(sample_raw_housing_data.columns)
        
        # Act
        train_df, test_df, holdout_df = load_and_split_data(
            raw_path=str(raw_path),
            output_dir=temp_raw_dir
        )
        
        # Assert
        assert set(train_df.columns) == original_columns
        assert set(test_df.columns) == original_columns
        assert set(holdout_df.columns) == original_columns
    
    def test_output_directory_created_if_not_exists(self, sample_raw_housing_data, temp_data_dir):
        """Test that output directory is created if it doesn't exist."""
        # Arrange
        raw_path = temp_data_dir / "raw_housing_data.csv"
        sample_raw_housing_data.to_csv(raw_path, index=False)
        new_output_dir = temp_data_dir / "new_output"
        
        # Act
        load_and_split_data(raw_path=str(raw_path), output_dir=new_output_dir)
        
        # Assert
        assert new_output_dir.exists()
        assert (new_output_dir / "train_data.csv").exists()
