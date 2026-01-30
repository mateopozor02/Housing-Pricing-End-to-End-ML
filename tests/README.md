# Tests

This directory contains comprehensive tests for the feature pipeline modules.

## Test Structure

```
tests/
├── conftest.py                    # Shared fixtures and test configuration
├── test_load.py                   # Tests for load.py module
├── test_preprocess.py             # Tests for preprocess.py module
└── test_feature_engineering.py    # Tests for feature_engineering.py module
```

## Running Tests

### Run all tests
```bash
uv run pytest
```

### Run tests with coverage report
```bash
uv run pytest --cov=src --cov-report=html
```

### Run specific test file
```bash
uv run pytest tests/test_load.py
uv run pytest tests/test_preprocess.py
uv run pytest tests/test_feature_engineering.py
```

### Run specific test class or function
```bash
uv run pytest tests/test_load.py::TestLoadAndSplitData
uv run pytest tests/test_preprocess.py::TestNormalizeCityName::test_normalize_lowercase
```

### Run tests by marker
```bash
# Run only unit tests
uv run pytest -m unit

# Run only integration tests
uv run pytest -m integration

# Run only slow tests
uv run pytest -m slow
```

### Run tests with verbose output
```bash
uv run pytest -v
```

### Run tests and stop at first failure
```bash
uv run pytest -x
```

## Test Markers

- `@pytest.mark.unit`: Unit tests that test individual functions in isolation
- `@pytest.mark.integration`: Integration tests that test multiple components together
- `@pytest.mark.slow`: Tests that take longer to execute

## Test Coverage

After running tests with coverage, open `htmlcov/index.html` in your browser to view detailed coverage reports:

```bash
open htmlcov/index.html  # macOS
```

## Writing New Tests

When adding new tests:

1. Follow the Arrange-Act-Assert (AAA) pattern
2. Use descriptive test names that explain what is being tested
3. Use appropriate fixtures from `conftest.py`
4. Add docstrings to test functions
5. Group related tests in test classes
6. Use appropriate markers (@pytest.mark.unit, @pytest.mark.integration, etc.)

Example:
```python
@pytest.mark.unit
def test_function_behavior(fixture_name):
    """Test that the function does what it should."""
    # Arrange
    input_data = ...
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result == expected_value
```

## Fixtures

Common fixtures available in `conftest.py`:

- `sample_raw_housing_data`: Sample raw housing data for testing
- `sample_metros_data`: Sample metros data with lat/lng
- `sample_train_data`: Sample preprocessed training data
- `sample_test_data`: Sample preprocessed test data
- `temp_data_dir`: Temporary directory for test files
- `temp_raw_dir`: Temporary raw data directory
- `temp_processed_dir`: Temporary processed data directory
- `temp_models_dir`: Temporary models directory

## Continuous Integration

These tests are designed to run in CI/CD pipelines. Ensure all tests pass before merging pull requests.
