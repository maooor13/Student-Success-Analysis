import pytest
import pandas as pd
from data import apply_iqr_outlier_removal, infer_column_roles

@pytest.fixture
def mock_data():
    """Generates a DataFrame with 30 elements, including 3 outliers."""
    values = [
        1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 
        6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 
        11, 11, 12, 12, 13, 14, 15, 
        85, 90, 100  # These should be removed
    ]
    binary_col = [
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1
    ]
    return pd.DataFrame({'scores': values, 'binary_col': binary_col})

def test_remove_outliers_count(mock_data):
    cleaned_df = apply_iqr_outlier_removal(mock_data)
    # Assert: We expect 27 elements to remain (30 - 3 outliers)
    assert len(cleaned_df) == 27

def test_outliers_are_actually_removed(mock_data):
    cleaned_df = apply_iqr_outlier_removal(mock_data)
    
    # Assert: The high values should not be in the cleaned data
    outliers = [85, 90, 100]
    for val in outliers:
        assert val not in cleaned_df['scores'].values

def test_normal_values_remain(mock_data):
    cleaned_df = apply_iqr_outlier_removal(mock_data)
    
    # Assert: A normal value like 7 should still be there
    assert 7 in cleaned_df['scores'].values


def test_skips_low_cardinality(mock_data):
    # Even though 100 is mathematically an outlier, the rule says skip it
    # Setting min_unique=7 (default) means 'binary_col' (2 unique) should be ignored
    result = apply_iqr_outlier_removal(mock_data)

    # Assert: The row with 100 still exists because the column was skipped
    assert len(result) == 27 # We expect 27 because the function removes 3 of a different column
    assert 1 in result['binary_col'].values


def test_infer_column_roles(mock_data):
    result = infer_column_roles(mock_data, target="scores")
    assert 'binary_col' in result['binary'] and 'scores' in result['target']