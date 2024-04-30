import pandas as pd
import pytest


@pytest.fixture
def sample_data():
    """Fixture to provide sample DataFrame."""
    return pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50]})


@pytest.fixture
def expected_scaled_data_A():
    """Fixture to provide expected scaled data for specific columns."""
    return pd.DataFrame(
        {
            "A": [0.0, 0.25, 0.5, 0.75, 1.0],  # MinMax scaled values for column 'A'
            "B": [
                10,
                20,
                30,
                40,
                50,
            ],  # Unchanged values for column 'B' as it won't be scaled in this test case
        }
    )


@pytest.fixture
def expected_scaled_data_A_B():
    """Fixture to provide expected scaled data for specific columns."""
    return pd.DataFrame(
        {
            "A": [0.0, 0.25, 0.5, 0.75, 1.0],  # MinMax scaled values for column 'A'
            "B": [0.0, 0.25, 0.5, 0.75, 1.0],  # Scaled values for both columns
        }
    )
