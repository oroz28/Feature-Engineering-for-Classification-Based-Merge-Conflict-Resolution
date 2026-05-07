import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, RFE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def select_features_chi2(X_train, y_train, X_test, k):
    """
    Filter method: Chi-squared statistic between each feature and the target.
    Requires non-negative features, so we apply MinMax scaling first.
    """
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    selector = SelectKBest(score_func=chi2, k=k)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)

    return X_train_selected, X_test_selected, selector


def select_features_mutual_info(X_train, y_train, X_test, k):
    """
    Filter method: Mutual information between each feature and the target.
    Can capture non-linear relationships. Does not require scaling.
    """
    selector = SelectKBest(score_func=lambda X,
                           y: mutual_info_classif(X, y, random_state=42), k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    return X_train_selected, X_test_selected, selector


def select_features_rfe(model, X_train, y_train, X_test, k):
    """
    Wrapper method: Recursive Feature Elimination. Uses the provided model to recursively eliminate least important features until k remain.
    """
    heavy_types = (RandomForestClassifier, LogisticRegression)
    try:
        heavy_types = (*heavy_types, XGBClassifier)
    except ImportError:
        pass

    estimator = DecisionTreeClassifier(
        random_state=42) if isinstance(model, heavy_types) else model

    selector = RFE(estimator=estimator, n_features_to_select=k, step=1)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    return X_train_selected, X_test_selected, selector


def select_features_rf_importance(X_train, y_train, X_test, k):
    """
    Embedded method: Train a Random Forest and select the top k features based on feature importance.
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError(
            "select_features_rf_importance requires a DataFrame input to track feature names.")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    importances = model.feature_importances_
    top_k_indices = np.argsort(importances)[::-1][:k]

    selector = {
        'indices': top_k_indices,
        'feature_names': X_train.columns[top_k_indices].tolist(),
        'importances': importances[top_k_indices],
    }

    X_train_selected = X_train.iloc[:, top_k_indices]
    X_test_selected = X_test.iloc[:, top_k_indices]

    return X_train_selected, X_test_selected, selector


def get_selected_feature_names(selector, feature_names, method):
    """
    Utility to extract selected feature names from the selector object based on the method used.
    """
    if method in ('chi2', 'mutual_info'):
        return list(np.array(feature_names)[selector.get_support()])

    elif method == 'rfe':
        return list(np.array(feature_names)[selector.support_])

    elif method == 'rf_importance':
        return selector['feature_names']

    elif method == 'pca':
        return feature_names

    else:
        raise ValueError(f"Unknown method: {method}")

# Not used


def select_features_sfs(model, X_train, y_train, X_test, k, direction='forward'):
    """
    Sequential Feature Selection (forward o backward)
    """
    sfs = SequentialFeatureSelector(
        model,
        n_features_to_select=k,
        direction=direction,
        scoring='f1_macro',
        cv=3,
        n_jobs=-1,
    )

    X_train_selected = sfs.fit_transform(X_train, y_train)
    X_test_selected = sfs.transform(X_test)

    return X_train_selected, X_test_selected, sfs


def select_features_embedded(X_train, y_train, X_test, C=0.1, l1_ratio=None):
    """
    Embedded method: Logistic Regression with L1 (Lasso) regularization. Selects features with non-zero coefficients.
    C is the inverse of regularization strength; smaller values specify stronger regularization.
    l1_ratio is the mix of L1 and L2 (if using elastic net).
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        solver='saga',
        C=C,
        l1_ratio=l1_ratio,
        max_iter=5000,
        random_state=42,
    )

    selector = SelectFromModel(model)

    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)

    return X_train_selected, X_test_selected, selector
