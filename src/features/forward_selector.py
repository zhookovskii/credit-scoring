import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import statsmodels.api as sm

class ForwardSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names, threshold=1e-2):
        self.feature_names_in_ = feature_names
        self.threshold = threshold
        self.support = None

    def fit(self, X, y):
        if isinstance(X, np.ndarray):
            if hasattr(self, "feature_names_in_"):
                X = pd.DataFrame(X, columns=self.feature_names_in_)
            else:
                raise ValueError("Feature names not provided")
        else:
            self.feature_names_in_ = X.columns.tolist()

        self.support = self._forward_stepwise_selection(X, y)
        return self

    def transform(self, X):
        return X[:, self.support]
    
    def get_support(self):
        if self.support is None:
            raise RuntimeError("Not fitted")
        return self.support
    
    def get_feature_names_out(self, input_features=None):
        features = np.array(self.feature_names_in_)
        return features[self.support]

    def _forward_stepwise_selection(self, X, y):
        included = {}
        y = list(y)

        while True:
            changed = False
            excluded = list(set(X.columns) - set(included.keys()))
            new_pvalues = pd.Series(index=excluded, dtype=float)

            for new_col in excluded:
                model = sm.Logit(y, sm.add_constant(X[list(included.keys()) + [new_col]])).fit(disp=0)
                new_pvalues[new_col] = model.pvalues[new_col]

            best_pval = new_pvalues.min()
            if best_pval < self.threshold:
                best_feature = new_pvalues.idxmin()
                included[best_feature] = True
                changed = True

            if not changed:
                break

        return np.array(
            [included.get(f, False) for f in X.columns]
        )