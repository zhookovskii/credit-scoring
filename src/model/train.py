import joblib
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from data.make_dataset import load_raw_data, clean_application_data, add_ratio_features
from features.make_features import load_feature_config, build_preprocessor
from features.forward_selector import ForwardSelector
from util.util import transform_omit_model


CONFIG_PATH = "src/config.yaml"
DATA_PATH = "data/raw/application_train.csv"


def load_features():
    config = load_feature_config(CONFIG_PATH)
    target = config["target"]
    num_features = config["numeric_features"]
    cat_features = config["categorical_features"]
    
    return num_features, cat_features, target


def build_pipe(num_features, cat_features):
    preprocessor = build_preprocessor(num_features, cat_features)

    selector = ForwardSelector(
        feature_names=num_features + cat_features
    )

    clf = Pipeline([
        ("preprocessor", preprocessor),
        ("selector", selector),
        ("model", LogisticRegression(max_iter=1000))
    ])

    return clf


def print_metrics(y_true, y_pred):
    roc_auc = roc_auc_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    ks = np.max(tpr - fpr)
    print(
        f'ROC AUC = {roc_auc}',
        f'PR_AUC = {pr_auc}',
        f'KS = {ks}',
        sep='\n'
    )


def build_explainer(clf, X_train):
    X_train = transform_omit_model(clf, X_train)

    return shap.LinearExplainer(
        model=clf.named_steps["model"],
        masker=X_train
    )


def main():
    df = load_raw_data(DATA_PATH)
    df = clean_application_data(df)
    df = add_ratio_features(df)

    num_features, cat_features, target = load_features()
    feature_list = num_features + cat_features

    X = df[feature_list]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = build_pipe(num_features, cat_features)
    clf.fit(X_train, y_train)

    print(f'Selected features: {X.columns[clf.named_steps["selector"].get_support()]}')

    prediction = clf.predict_proba(X_train)[:, 1]
    print("Train metrics:")
    print_metrics(y_train, prediction)

    prediction = clf.predict_proba(X_test)[:, 1]
    print("Test metrics:")
    print_metrics(y_test, prediction)

    explainer = build_explainer(clf, X_train)

    joblib.dump(clf, "artifacts/logreg_pipeline.pkl")
    joblib.dump(explainer, "artifacts/shap_explainer.pkl")


if __name__ == "__main__":
    main()
