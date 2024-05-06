from classification_model.predict import make_prediction


def test_make_prediction(sample_input, expected_output):
    """Test the make_prediction function with sample input and a mock prediction model."""
    result = make_prediction(input_data=sample_input)

    assert (
        result["prediction"] == expected_output
    ), "Expected prediction does not match."
    assert isinstance(result["prediction"], list)
    assert all(
        x in [0, 1] for x in result["prediction"]
    ), "Prediction should only contain 0 or 1."
    assert result["errors"] is None, "Expected no validation errors."
