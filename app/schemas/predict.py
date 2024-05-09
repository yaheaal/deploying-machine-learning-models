from typing import Any, ClassVar, List, Optional

from pydantic import BaseModel, ConfigDict

from classification_model.processing.validation import PassengerDataInputSchema


class ClassificationResults(BaseModel):
    errors: Optional[Any]
    version: str
    prediction: Optional[List[float]]


class MultiplePassengerDataInputs(BaseModel):
    inputs: List[PassengerDataInputSchema]

    config: ClassVar[ConfigDict] = ConfigDict(
        json_schema_extra={
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
                        "Cabin": None,
                        "Embarked": "S",
                    }
                ]
            }
        }
    )
