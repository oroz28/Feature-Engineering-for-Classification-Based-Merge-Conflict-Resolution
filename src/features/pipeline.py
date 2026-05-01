from src.features.selectors import (
    select_features_chi2,
    select_features_mutual_info,
    select_features_rfe,
    select_features_sfs,
    select_features_lasso,
    select_features_rf_importance
)

def feature_selection_pipeline(X_train, y_train, X_test, method, model=None, k=None):
    """
    Pipeline for feature selection based on the specified method.
    """
    if method in ['rfe', 'sfs', 'sbs', 'lasso'] and model is None:
        raise ValueError(f"Model must be provided for method {method}")

    if method not in ['lasso'] and k is None:
        raise ValueError(f"k must be provided for method {method}")

    if method == 'chi2':
        return select_features_chi2(X_train, y_train, X_test, k)

    elif method == 'mutual_info':
        return select_features_mutual_info(X_train, y_train, X_test, k)

    elif method == 'rfe':
        return select_features_rfe(model, X_train, y_train, X_test, k)

    elif method in ['sfs', 'sbs']:
        direction = 'forward' if method == 'sfs' else 'backward'
        return select_features_sfs(model, X_train, y_train, X_test, k, direction)

    elif method == 'lasso':
        return select_features_lasso(model, X_train, y_train, X_test)

    elif method == 'rf_importance':
        return select_features_rf_importance(X_train, y_train, X_test, k)

    else:
        raise ValueError(f"Method {method} not supported")