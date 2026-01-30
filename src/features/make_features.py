import yaml
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from features.woe_encoder import WOEEncoder


def load_feature_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown"))
    ])

    col_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    features = numeric_features + categorical_features
    woe_encoder = WOEEncoder(
        feature_names = features, 
        categorical_features=categorical_features
    )

    preprocessor = Pipeline([
        ("column_transformer", col_transformer),
        ("woe", woe_encoder),
        ("scaler", StandardScaler())
    ])

    return preprocessor
