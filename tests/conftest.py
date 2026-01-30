"""
Pytest configuration and shared fixtures for feature pipeline tests.
"""

import pandas as pd
import pytest
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def sample_raw_housing_data():
    """Create sample raw housing data for testing."""
    return pd.DataFrame({
        "date": ["2019-01-01", "2019-06-01", "2020-06-01", "2021-01-01", "2022-06-01"],
        "city_full": ["New York-Newark-Jersey City", "Los Angeles-Long Beach-Anaheim", 
                      "Chicago-Naperville-Elgin", "DC_Metro", "Boston-Cambridge-Newton"],
        "city": ["New York", "Los Angeles", "Chicago", "Washington", "Boston"],
        "zipcode": ["10001", "90001", "60601", "20001", "02101"],
        "median_list_price": [500000, 600000, 350000, 450000, 550000],
        "median_sale_price": [480000, 580000, 340000, 440000, 540000],
    })


@pytest.fixture
def sample_metros_data():
    """Create sample metros data for testing."""
    return pd.DataFrame({
        "metro_full": [
            "new york-newark-jersey city, ny-nj",
            "los angeles-long beach-anaheim, ca",
            "chicago-naperville-elgin, il-in",
            "washington-arlington-alexandria, dc-va-md-wv",
            "boston-cambridge-newton, ma-nh"
        ],
        "lat": [40.7128, 34.0522, 41.8781, 38.9072, 42.3601],
        "lng": [-74.0060, -118.2437, -87.6298, -77.0369, -71.0589]
    })


@pytest.fixture
def sample_train_data():
    """Create sample training data after preprocessing."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2019-01-01", "2019-02-01", "2019-03-01"]),
        "city_full": ["new york-newark-jersey city, ny-nj", "los angeles-long beach-anaheim, ca", 
                      "chicago-naperville-elgin, il-in"],
        "city": ["new york", "los angeles", "chicago"],
        "zipcode": ["10001", "90001", "60601"],
        "median_list_price": [500000, 600000, 350000],
        "median_sale_price": [480000, 580000, 340000],
        "lat": [40.7128, 34.0522, 41.8781],
        "lng": [-74.0060, -118.2437, -87.6298]
    })


@pytest.fixture
def sample_test_data():
    """Create sample test data after preprocessing."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2020-06-01", "2020-07-01"]),
        "city_full": ["new york-newark-jersey city, ny-nj", "los angeles-long beach-anaheim, ca"],
        "city": ["new york", "los angeles"],
        "zipcode": ["10001", "90001"],
        "median_list_price": [520000, 620000],
        "median_sale_price": [500000, 600000],
        "lat": [40.7128, 34.0522],
        "lng": [-74.0060, -118.2437]
    })


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_raw_dir(temp_data_dir):
    """Create a temporary raw data directory."""
    raw_dir = temp_data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir


@pytest.fixture
def temp_processed_dir(temp_data_dir):
    """Create a temporary processed data directory."""
    processed_dir = temp_data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir


@pytest.fixture
def temp_models_dir(temp_data_dir):
    """Create a temporary models directory."""
    models_dir = temp_data_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir
