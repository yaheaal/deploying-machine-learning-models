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


@pytest.fixture
def sample_input():
    """Provides sample input data for testing the prediction function."""
    data = {
        "PassengerId": [1, 2, 3, 4, 5],
        "Pclass": [3, 1, 3, 1, 3],
        "Name": [
            "Braund, Mr. Owen Harris",
            "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
            "Heikkinen, Miss. Laina",
            "Futrelle, Mrs. Jacques Heath (Lily May Peel)",
            "Allen, Mr. William Henry",
        ],
        "Sex": ["male", "female", "female", "female", "male"],
        "Age": [22.0, 38.0, 26.0, 35.0, 35.0],
        "SibSp": [1, 1, 0, 1, 0],
        "Parch": [0, 0, 0, 0, 0],
        "Ticket": ["A/5 21171", "PC 17599", "STON/O2. 3101282", "113803", "373450"],
        "Fare": [7.2500, 71.2833, 7.9250, 53.1000, 8.0500],
        "Cabin": [None, "C85", None, "C123", None],
        "Embarked": ["S", "C", "S", "S", "S"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def expected_output():
    """Fixture to provide expected output for the predictions."""
    return [0, 1, 1, 1, 0]
