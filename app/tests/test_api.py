import numpy as np


def test_api(client, sample_input, expected_output):
    # Given
    payload = {"inputs": sample_input.replace({np.nan: None}).to_dict(orient="records")}

    # When
    response = client.post(
        "http://localhost:8000/api/v1/predict",
        json=payload,
    )

    # Then
    assert response.status_code == 200
    prediction_data = response.json()
    assert isinstance(prediction_data["prediction"], list)
    assert prediction_data["errors"] is None, "Expected no validation errors."
    assert (
        prediction_data["prediction"] == expected_output
    ), "Expected prediction does not match."
