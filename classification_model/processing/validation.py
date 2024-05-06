from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    validated_data = input_data.replace(
        {np.nan: None}
    )  # Replace numpy NaN with None for Pydantic validation
    errors = None

    try:
        # Print the dictionaries to see if keys are strings
        records = validated_data.to_dict(orient="records")
        inputs = [
            PassengerDataInputSchema(
                **{str(key): value for key, value in record.items()}
            )
            for record in records
        ]
        MultiplePassengerDataInputs(inputs=inputs)
    except ValidationError as error:
        errors = {
            e["loc"][0]: e["msg"] for e in error.errors()
        }  # Formatting errors as a dictionary

    return validated_data, errors


class PassengerDataInputSchema(BaseModel):
    PassengerId: Optional[int]
    Pclass: Optional[int]
    Name: Optional[str]
    Sex: Optional[str]
    Age: Optional[float]
    SibSp: Optional[int]
    Parch: Optional[int]
    Ticket: Optional[str]
    Fare: Optional[float]
    Cabin: Optional[str]
    Embarked: Optional[str]


class MultiplePassengerDataInputs(BaseModel):
    inputs: List[PassengerDataInputSchema]
