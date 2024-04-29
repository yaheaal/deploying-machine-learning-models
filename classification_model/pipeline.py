from category_encoders.ordinal import OrdinalEncoder
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from feature_engine.selection import DropFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import classification_model.processing.feature as pp
from classification_model.config.core import config

mapping_var_dicts = [
    {"col": item.col, "mapping": item.mapping} for item in config.model.mapping_var
]

preprocessing_pipeline = Pipeline(
    [
        # ====== DROP ======
        # Drop unnecessary features
        (
            "drop_unnecessary",
            DropFeatures(features_to_drop=config.model.drop_variables),
        ),
        # ====== IMPUTER ======
        # Impute categprocal data
        (
            "frequent_imputer",
            CategoricalImputer(
                imputation_method="frequent", variables=config.model.cat_na_with_mode
            ),
        ),
        # Impute numerical data
        (
            "median_imputer",
            MeanMedianImputer(
                imputation_method="median", variables=config.model.num_na_with_median
            ),
        ),
        # ====== CUT ======
        # Cut skewness data
        (
            "cut_skewness",
            pp.FareDiscretizer(
                bins=config.model.bins_fare, labels=config.model.labels_fare
            ),
        ),
        # ======  MAPPING ======
        # Apply mapping
        (
            "Map_categorical",
            OrdinalEncoder(
                mapping=mapping_var_dicts,
            ),
        ),
        # ====== DUMMIES ======
        # Apply dummies
        (
            "Encode_categorical",
            pp.CustomOneHotEncoder(
                drop_cols=config.model.one_hot_drop, columns=config.model.one_hot_var
            ),
        ),
        # ====== SCALING ======
        # Feature Scaling
        ("Scale_features", pp.CustomScaler(columns=["Age"])),
        # ====== SELECTION ======
        ("Select_features", pp.FeatureSelector(columns=config.model.selected_features)),
    ]
)

model_pipeline = Pipeline(
    [
        # ====== MODEL ======
        ("train_model", LogisticRegression(random_state=config.model.random_state))
    ]
)

main_pipeline = Pipeline(
    [("preprocessing", preprocessing_pipeline), ("modeling", model_pipeline)]
)
