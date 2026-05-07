""" 
feature_engineering.py:
Module for feature engineering techniques: scaling, polynomial features, and PCA.
These are applied as pre-processing transformations before or instead of feature selection.

Key design decisions:
- All transformers are fit on X_train only and applied to X_test (no leakage).
- PCA is treated as a dimensionality reduction alternative to feature selection,
  not as a pre-step before selection (combining both would double-reduce).
- Polynomial features are only practical on a small subset of features due to O(k^2) expansion.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.decomposition import PCA


def apply_scaling(X_train, X_test, method='standard'):
    """
    Apply feature scaling. Fit on train, transform both.

    Parameters
    ----------
    method : 'standard' (zero mean, unit variance) or 'minmax' ([0, 1])
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


def apply_polynomial(X_train, X_test, degree=2, interaction_only=True):
    """
    Polynomial feature expansion. Use interaction_only=True to avoid x^2 terms
    and keep the feature count manageable: C(k, 2) instead of C(k+2, 2).

    Warning: only apply on a pre-selected subset of features. With k=43 and
    interaction_only=False, degree=2 produces 946 features — too many for most models.
    With interaction_only=True and k=10: 55 features — reasonable.

    Parameters
    ----------
    interaction_only : bool
        If True, only interaction terms (no x_i^2). Recommended for this use case.
    """
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    feature_names = poly.get_feature_names_out()
    return X_train_poly, X_test_poly, poly, feature_names


def apply_pca(X_train, X_test, n_components=0.95, scale_first=True):
    """
    PCA-based dimensionality reduction. Alternative to feature selection —
    transforms features into orthogonal components ordered by explained variance.

    Parameters
    ----------
    n_components : float or int
        If float in (0, 1): keep enough components to explain that fraction of variance.
        If int: keep exactly that many components.
    scale_first : bool
        PCA requires standardized data. If True, applies StandardScaler before PCA.
        Set False only if data is already scaled.

    Returns
    -------
    X_train_pca, X_test_pca, pca, n_components_used, explained_variance_ratio
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
    Fit full PCA (all components) and return cumulative explained variance.
    Used to determine how many components capture 90%, 95%, 99% of variance.

    This is a standalone analysis function — does NOT split train/test,
    so call it on the full feature matrix for exploratory analysis only.

    Returns
    -------
    dict with:
        - cumulative_variance: array of cumulative explained variance ratios
        - n_components_90/95/99: components needed for each threshold
        - explained_variance_ratio: per-component ratios
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