import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.decomposition import PCA


def apply_scaling(X_train, X_test, method='standard'):
    """
    Apply feature scaling. Fit on train, transform both.
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("method must be 'standard' or 'minmax'")

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def apply_pca(X_train, X_test, n_components=0.95, scale_first=True):
    """
    Apply PCA for dimensionality reduction. Fit on train, transform both.
    n_components can be int (number of components) or float (variance threshold).
    """
    if scale_first:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_test_pca, pca


def pca_variance_analysis(X, scale_first=True):
    """
    Fits PCA on the entire feature set and analyzes explained variance.
    """
    if isinstance(X, pd.DataFrame):
        X = X.values

    if scale_first:
        X = StandardScaler().fit_transform(X)

    pca = PCA(random_state=42)
    pca.fit(X)

    cumvar = np.cumsum(pca.explained_variance_ratio_)

    def components_for_threshold(threshold):
        idx = np.argmax(cumvar >= threshold)
        return int(idx + 1)

    return {
        "cumulative_variance": cumvar,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "n_components_90": components_for_threshold(0.90),
        "n_components_95": components_for_threshold(0.95),
        "n_components_99": components_for_threshold(0.99),
        "total_features": X.shape[1],
    }
