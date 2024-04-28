from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel

import classification_model

PACKAGE_ROOT = Path(classification_model.__file__).resolve().parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"


class AppConfig(BaseModel):
    """
    Application-level config.
    """

    train_data_file: str
    test_data_file: str
    package_name: str
    pipeline_name: str
    pipeline_save_file: str


class MappingItem(BaseModel):
    col: str
    mapping: Dict[str, Union[int, str]]


class ModelConfig(BaseModel):
    """
    All configration relevant to model
    traning and feature engineering.
    """

    test_size: float
    random_state: int
    target: str
    drop_variables: List[str]
    cat_na_with_mode: List[str]
    num_na_with_median: List[str]
    bins_fare: List[float]
    labels_fare: List[str]
    mapping_var: List[MappingItem]
    one_hot_var: List[str]
    one_hot_drop: List[str]
    selected_features: List[str]


class Config(BaseModel):
    """
    Master config object.
    """

    app: AppConfig
    model: ModelConfig


def find_config_file() -> Path:
    """
    Locate the configuration file.
    """

    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise FileNotFoundError(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Parse YAML containing the package configuration.
    """

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as config_file:
            parsed_config = yaml.safe_load(config_file)
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(
    parsed_config: Optional[Dict[str, Any]] = None
) -> Config:
    """
    Run validation on config values.
    """

    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    app_config_data = parsed_config.get("app", {})
    model_config_data = parsed_config.get("model", {})

    return Config(
        app=AppConfig(**app_config_data), model=ModelConfig(**model_config_data)
    )


config = create_and_validate_config()
