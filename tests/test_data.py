import pytest
import pandas as pd
from data import apply_iqr_outlier_removal

@pytest.fixture
def mock_data():
    """Generates a DataFrame with 30 elements, including 3 outliers."""
    values = [
        1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 
        6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 
        11, 11, 12, 12, 13, 14, 15, 
        85, 90, 100  # These should be removed
    ]
    return pd.DataFrame({'scores': values})

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