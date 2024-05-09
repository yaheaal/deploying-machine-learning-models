import json
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from loguru import logger

from app import schemas
from app.config import settings
from app.version import __version__
from classification_model import __version__ as m_version
from classification_model.predict import make_prediction

api_router = APIRouter()


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, version_model=m_version
    )

    return health.dict()


@api_router.post(
    "/predict", response_model=schemas.ClassificationResults, status_code=200
)
async def predict(input_data: schemas.MultiplePassengerDataInputs) -> Any:
    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))

    logger.info(f"Making prediction on inputs: {input_data.inputs}")
    results = make_prediction(input_data=input_df)

    if results["errors"] is not None:
        logger.warning(f"Prediction validation error: {results.get('errors')}")
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    logger.info(f"Prediction results: {results.get('prediction')}")

    return results
