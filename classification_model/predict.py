import typing as t

import pandas as pd

from classification_model import __version__ as _version
from classification_model.config.core import config
from classification_model.processing.data_manager import load_pipeline
from classification_model.processing.validation import validate_inputs

pipeline_file_name = f"{config.app.pipeline_save_file}{_version}.pkl"
_price_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(
    *,
    input_data: t.Union[pd.DataFrame, dict],
) -> dict:

    data = pd.DataFrame(input_data)
    validate_data, errors = validate_inputs(input_data=data)
    result = {"prediction": None, "version": _version, "errors": errors}

    if not errors:
        prediction = _price_pipe.predict(X=validate_data)
        result["prediction"] = prediction.tolist()
    return result
