from typing import Any, List, Optional

from pydantic import BaseModel

from classification_model.processing.validation import PassengerDataInputSchema


class ClassificationResults(BaseModel):
    errors: Optional[Any]
    version: str
    prediction: Optional[List[float]]


class MultiplePassengerDataExample(BaseModel):
    inputs: List[PassengerDataInputSchema]

    class Config:
        json_schema_extra = {
            "example": {
                "inputs": [
                    {
                        "PassengerId": 1,
                        "Pclass": 3,
                        "Name": "Braund, Mr. Owen Harris",
                        "Sex": "male",
                        "Age": 22.0,
                        "SibSp": 1,
                        "Parch": 0,
                        "Ticket": "A/5 21171",
                        "Fare": 7.25,
                        "Cabin": "C85",
                        "Embarked": "S",
                    }
                ]
            }
        }
