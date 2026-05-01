from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.decomposition import PCA

def apply_scaling(X_train, X_test, method='standard'):
    """
    Apply scaling to the features.
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


def apply_polynomial(X_train, X_test, degree=2, interaction_only=False):
    """
    Apply polynomial feature expansion.
    """
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    return X_train_poly, X_test_poly, poly


def apply_pca(X_train, X_test, variance=0.95):
    """
    Apply PCA to reduce dimensionality.
    variance: float, percentage of variance to keep (0-1)
    """
    pca = PCA(n_components=variance)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca