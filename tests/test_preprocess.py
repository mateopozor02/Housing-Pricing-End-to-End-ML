"""
Unit tests for the preprocess module (data cleaning and preprocessing).
"""

import pandas as pd
import pytest
from pathlib import Path
from src.feature_pipeline.preprocess import (
    normalize_city_name,
    clean_and_merge_city_names,
    remove_duplicatates,
    remove_outliers,
    preprocess_data_split,
    run_preprocessing_pipeline,
    CITY_MAPPING
)


@pytest.mark.unit
class TestNormalizeCityName:
    """Test suite for normalize_city_name function."""
    
    def test_normalize_lowercase(self):
        """Test that city names are converted to lowercase."""
        assert normalize_city_name("NEW YORK") == "new york"
        assert normalize_city_name("Los Angeles") == "los angeles"
    
    def test_normalize_strips_whitespace(self):
        """Test that leading/trailing whitespace is removed."""
        assert normalize_city_name("  Boston  ") == "boston"
        assert normalize_city_name("Chicago ") == "chicago"
    
    def test_normalize_standardizes_dashes(self):
        """Test that different dash types are standardized."""
        assert normalize_city_name("New York – Newark") == "new york-newark"
        assert normalize_city_name("San Francisco  -  Oakland") == "san francisco-oakland"
    
    def test_normalize_removes_extra_spaces(self):
        """Test that multiple spaces are collapsed to single spaces."""
        assert normalize_city_name("Los   Angeles") == "los angeles"
    
    def test_normalize_handles_nan(self):
        """Test that NaN values are handled correctly."""
        result = normalize_city_name(pd.NA)
        assert pd.isna(result)
    
    def test_normalize_handles_none(self):
        """Test that None values are handled correctly."""
        result = normalize_city_name(None)
        assert pd.isna(result)


@pytest.mark.unit
class TestCleanAndMergeCityNames:
    """Test suite for clean_and_merge_city_names function."""
    
    def test_replaces_non_standard_city_names(self):
        """Test that non-standard city names are replaced using CITY_MAPPING."""
        # Arrange
        df = pd.DataFrame({
            "city_full": ["DC_Metro", "Pittsburgh"],
            "median_list_price": [500000, 300000]
        })
        
        # Act
        result = clean_and_merge_city_names(df, metros_path=None)
        
        # Assert
        assert result.loc[0, "city_full"] == CITY_MAPPING["DC_Metro"]
        assert result.loc[1, "city_full"] == CITY_MAPPING["Pittsburgh"]
    
    def test_merges_lat_lng_from_metros(self, sample_metros_data, temp_raw_dir):
        """Test that lat/lng coordinates are merged from metros data."""
        # Arrange
        metros_path = temp_raw_dir / "metros.csv"
        sample_metros_data.to_csv(metros_path, index=False)
        
        # Use lowercase city names to match metros data
        df = pd.DataFrame({
            "city_full": ["new york-newark-jersey city, ny-nj", "los angeles-long beach-anaheim, ca"],
            "median_list_price": [500000, 600000]
        })
        
        # Act
        result = clean_and_merge_city_names(df, metros_path=str(metros_path))
        
        # Assert
        assert "lat" in result.columns
        assert "lng" in result.columns
        assert result.loc[0, "lat"] == 40.7128
        assert result.loc[0, "lng"] == -74.0060
    
    def test_skips_merge_if_lat_lng_exist(self, sample_metros_data, temp_raw_dir):
        """Test that merge is skipped if lat/lng columns already exist."""
        # Arrange
        metros_path = temp_raw_dir / "metros.csv"
        sample_metros_data.to_csv(metros_path, index=False)
        
        df = pd.DataFrame({
            "city_full": ["New York-Newark-Jersey City, NY-NJ"],
            "lat": [99.9999],
            "lng": [-99.9999],
            "median_list_price": [500000]
        })
        
        # Act
        result = clean_and_merge_city_names(df, metros_path=str(metros_path))
        
        # Assert - original lat/lng should be preserved
        assert result.loc[0, "lat"] == 99.9999
        assert result.loc[0, "lng"] == -99.9999
    
    def test_handles_missing_metros_file(self):
        """Test that function handles missing metros file gracefully."""
        # Arrange
        df = pd.DataFrame({
            "city_full": ["New York-Newark-Jersey City, NY-NJ"],
            "median_list_price": [500000]
        })
        
        # Act
        result = clean_and_merge_city_names(df, metros_path="nonexistent.csv")
        
        # Assert - should return df without lat/lng columns
        assert "lat" not in result.columns
        assert "lng" not in result.columns
    
    def test_handles_missing_city_full_column(self):
        """Test that function handles missing city_full column."""
        # Arrange
        df = pd.DataFrame({
            "city": ["New York"],
            "median_list_price": [500000]
        })
        
        # Act
        result = clean_and_merge_city_names(df, metros_path=None)
        
        # Assert - should return df unchanged
        assert "city_full" not in result.columns
        pd.testing.assert_frame_equal(result, df)


@pytest.mark.unit
class TestRemoveDuplicates:
    """Test suite for remove_duplicatates function."""
    
    def test_removes_duplicate_rows(self):
        """Test that duplicate rows are removed."""
        # Arrange
        df = pd.DataFrame({
            "city_full": ["New York", "New York", "Los Angeles"],
            "zipcode": ["10001", "10001", "90001"],
            "date": ["2019-01-01", "2019-02-01", "2019-01-01"],
            "median_list_price": [500000, 500000, 600000]
        })
        
        # Act
        result = remove_duplicatates(df)
        
        # Assert - duplicate should be removed (based on all columns except date)
        assert len(result) < len(df)
    
    def test_excludes_date_year_from_duplicate_check(self):
        """Test that date and year columns are excluded from duplicate detection."""
        # Arrange
        df = pd.DataFrame({
            "city_full": ["New York", "New York"],
            "zipcode": ["10001", "10001"],
            "date": ["2019-01-01", "2019-02-01"],
            "year": [2019, 2019],
            "median_list_price": [500000, 500000]
        })
        
        # Act
        result = remove_duplicatates(df)
        
        # Assert - rows should be considered duplicates despite different dates
        assert len(result) == 0  # Both rows removed when keep=False
    
    def test_keeps_unique_rows(self):
        """Test that unique rows are preserved."""
        # Arrange
        df = pd.DataFrame({
            "city_full": ["New York", "Los Angeles", "Chicago"],
            "zipcode": ["10001", "90001", "60601"],
            "median_list_price": [500000, 600000, 350000]
        })
        original_len = len(df)
        
        # Act
        result = remove_duplicatates(df)
        
        # Assert
        assert len(result) == original_len


@pytest.mark.unit
class TestRemoveOutliers:
    """Test suite for remove_outliers function."""
    
    def test_removes_prices_above_threshold(self):
        """Test that outliers above 19M are removed."""
        # Arrange
        df = pd.DataFrame({
            "city_full": ["New York", "Expensive City", "Los Angeles"],
            "median_list_price": [500000, 20000000, 600000]
        })
        
        # Act
        result = remove_outliers(df)
        
        # Assert
        assert len(result) == 2
        assert all(result["median_list_price"] <= 19000000)
    
    def test_keeps_prices_at_threshold(self):
        """Test that prices exactly at 19M threshold are kept."""
        # Arrange
        df = pd.DataFrame({
            "city_full": ["City1", "City2"],
            "median_list_price": [19000000, 18999999]
        })
        
        # Act
        result = remove_outliers(df)
        
        # Assert
        assert len(result) == 2
    
    def test_keeps_all_valid_prices(self):
        """Test that all valid prices are preserved."""
        # Arrange
        df = pd.DataFrame({
            "city_full": ["New York", "Los Angeles", "Chicago"],
            "median_list_price": [500000, 600000, 350000]
        })
        original_len = len(df)
        
        # Act
        result = remove_outliers(df)
        
        # Assert
        assert len(result) == original_len


@pytest.mark.integration
class TestPreprocessDataSplit:
    """Integration tests for preprocess_data_split function."""
    
    def test_preprocesses_single_split(self, sample_raw_housing_data, temp_raw_dir, temp_processed_dir):
        """Test that a single data split is preprocessed correctly."""
        # Arrange
        split_name = "train"
        input_path = temp_raw_dir / f"{split_name}_data.csv"
        sample_raw_housing_data.to_csv(input_path, index=False)
        
        # Act
        result = preprocess_data_split(
            split=split_name,
            raw_dir=temp_raw_dir,
            processed_dir=temp_processed_dir,
            metros_path=None
        )
        
        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= len(sample_raw_housing_data)  # May be less due to cleaning
        output_path = temp_processed_dir / f"cleaning_{split_name}_data.csv"
        assert output_path.exists()
    
    def test_output_file_naming(self, sample_raw_housing_data, temp_raw_dir, temp_processed_dir):
        """Test that output file has correct naming convention."""
        # Arrange
        split_name = "test"
        input_path = temp_raw_dir / f"{split_name}_data.csv"
        sample_raw_housing_data.to_csv(input_path, index=False)
        
        # Act
        preprocess_data_split(
            split=split_name,
            raw_dir=temp_raw_dir,
            processed_dir=temp_processed_dir,
            metros_path=None
        )
        
        # Assert
        expected_output = temp_processed_dir / f"cleaning_{split_name}_data.csv"
        assert expected_output.exists()


@pytest.mark.integration
class TestRunPreprocessingPipeline:
    """Integration tests for run_preprocessing_pipeline function."""
    
    def test_processes_all_splits(self, sample_raw_housing_data, temp_raw_dir, temp_processed_dir):
        """Test that all data splits are processed."""
        # Arrange
        splits = ("train", "test", "holdout")
        for split in splits:
            input_path = temp_raw_dir / f"{split}_data.csv"
            sample_raw_housing_data.to_csv(input_path, index=False)
        
        # Act
        run_preprocessing_pipeline(
            splits=splits,
            raw_dir=temp_raw_dir,
            processed_dir=temp_processed_dir,
            metros_path=None
        )
        
        # Assert
        for split in splits:
            output_path = temp_processed_dir / f"cleaning_{split}_data.csv"
            assert output_path.exists()
    
    def test_processes_custom_splits(self, sample_raw_housing_data, temp_raw_dir, temp_processed_dir):
        """Test that custom split names work correctly."""
        # Arrange
        custom_splits = ("train", "validation")
        for split in custom_splits:
            input_path = temp_raw_dir / f"{split}_data.csv"
            sample_raw_housing_data.to_csv(input_path, index=False)
        
        # Act
        run_preprocessing_pipeline(
            splits=custom_splits,
            raw_dir=temp_raw_dir,
            processed_dir=temp_processed_dir,
            metros_path=None
        )
        
        # Assert
        for split in custom_splits:
            output_path = temp_processed_dir / f"cleaning_{split}_data.csv"
            assert output_path.exists()
