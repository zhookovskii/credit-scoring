import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class WOEEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names, categorical_features):
        self.categorical_features = categorical_features
        self.feature_names_in_ = feature_names
        self.woe_maps = {}

    def fit(self, X, y):
        if isinstance(X, np.ndarray):
            if hasattr(self, "feature_names_in_"):
                X = pd.DataFrame(X, columns=self.feature_names_in_)
            else:
                raise ValueError("Feature names not provided")
        else:
            self.feature_names_in_ = X.columns.tolist()

        y_series = pd.Series(y, name="target")

        for col in self.categorical_features:
            woe_table = self._compute_woe(pd.concat([X[[col]], y_series], axis=1), col)
            self.woe_maps[col] = woe_table['woe'].to_dict()

        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            if hasattr(self, "feature_names_in_"):
                X = pd.DataFrame(X, columns=self.feature_names_in_)
            else:
                raise ValueError(
                    "X is numpy array but feature names not provided."
                )
        X = X.copy()
        for col in self.categorical_features:
            X[col] = X[col].map(self.woe_maps[col]).fillna(0.0)

        return X
    
    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_in_)

    def _compute_woe(self, df, feature):
        eps = 1e-6
        grouped = df.groupby(feature)['target'].agg(['count', 'sum'])
        grouped.columns = ['total', 'bad']
        grouped['good'] = grouped['total'] - grouped['bad']

        total_good = grouped['good'].sum()
        total_bad = grouped['bad'].sum()

        grouped['dist_good'] = grouped['good'] / total_good
        grouped['dist_bad'] = grouped['bad'] / total_bad

        grouped['woe'] = np.log((grouped['dist_good'] + eps) / (grouped['dist_bad'] + eps))
        return grouped
