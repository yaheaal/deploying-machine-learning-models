import pandas as pd

from classification_model.processing.feature import CustomScaler


def test_column_scaling(sample_data, expected_scaled_data_A):
    """Test that only specified columns are scaled."""
    scaler = CustomScaler(columns=["A"])  # Only scale column 'A'
    scaled_data = scaler.fit_transform(sample_data)

    pd.testing.assert_frame_equal(scaled_data, expected_scaled_data_A)


def test_full_dataframe_scaling(sample_data, expected_scaled_data_A_B):
    """Test that the entire DataFrame is scaled if no columns are specified."""
    scaler = CustomScaler()  # Scale all columns
    scaled_data = scaler.fit_transform(sample_data)

    pd.testing.assert_frame_equal(scaled_data, expected_scaled_data_A_B)
