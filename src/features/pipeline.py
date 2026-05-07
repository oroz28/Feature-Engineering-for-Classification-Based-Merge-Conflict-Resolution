from src.features.selectors import (
    select_features_chi2,
    select_features_mutual_info,
    select_features_rfe,
    select_features_rf_importance,
)
from src.features.feature_engineering import apply_pca

METHODS_REQUIRING_K = {'chi2', 'mutual_info', 'rfe', 'rf_importance', 'poly'}
METHODS_REQUIRING_MODEL = {'rfe'}
SUPPORTED_METHODS = METHODS_REQUIRING_K | {'baseline', 'pca'}


def feature_selection_pipeline(X_train, y_train, X_test, method, model=None, k=None):
    """
    Applies the specified feature selection or dimensionality reduction method.
    Validates method and required parameters. Returns transformed train/test sets and the fitted selector object.
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Method '{method}' not supported. Choose from: {
                         sorted(SUPPORTED_METHODS)}")

    if method in METHODS_REQUIRING_MODEL and model is None:
        raise ValueError(f"Method '{method}' requires a model.")

    if method in METHODS_REQUIRING_K and k is None:
        raise ValueError(f"Method '{method}' requires k.")

    if k is not None and isinstance(k, int) and k >= X_train.shape[1]:
        raise ValueError(
            f"k={k} must be < number of features ({X_train.shape[1]}).")

    if method == 'chi2':
        return select_features_chi2(X_train, y_train, X_test, k)

    elif method == 'mutual_info':
        return select_features_mutual_info(X_train, y_train, X_test, k)

    elif method == 'rfe':
        return select_features_rfe(model, X_train, y_train, X_test, k)

    elif method == 'rf_importance':
        return select_features_rf_importance(X_train, y_train, X_test, k)

    elif method == 'pca':
        n_components = k if k is not None else 0.95
        X_train_out, X_test_out, pca = apply_pca(
            X_train, X_test, n_components=n_components, scale_first=True)
        return X_train_out, X_test_out, pca

    else:
        # Baseline: no feature selection, return original sets and None for selector
        return X_train, X_test, None
