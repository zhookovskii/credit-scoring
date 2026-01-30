def transform_omit_model(clf, X):
    X = clf.named_steps["preprocessor"].transform(X)
    X = clf.named_steps["selector"].transform(X)
    
    return X