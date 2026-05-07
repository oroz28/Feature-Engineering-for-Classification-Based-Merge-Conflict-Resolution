
# import numpy as np

# from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.feature_selection import RFE
# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_selection import SelectFromModel
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_selection import SequentialFeatureSelector


# def select_features_chi2(X_train, y_train, X_test, k):
#     """
#     Selects the k best features using Chi-square.
#     """
#     if (X_train < 0).sum().sum() > 0 or (X_test < 0).sum().sum() > 0:
#         print("Warning: Negative values found, applying MinMaxScaler.")
#         # scale to [0,1] (necessary for chi2)
#         scaler = MinMaxScaler()
    
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_test_scaled = scaler.transform(X_test)
    
#     else:
#         X_train_scaled = X_train.copy()
#         X_test_scaled = X_test.copy()
        
#     # feature selection
#     selector = SelectKBest(score_func=chi2, k=k)
    
#     X_train_selected = selector.fit_transform(X_train_scaled, y_train)
#     X_test_selected = selector.transform(X_test_scaled)    
#     # selected_features = X_train.columns[selector.get_support()]
    
#     return X_train_selected, X_test_selected, selector


# def select_features_mutual_info(X_train, y_train, X_test, k):
#     """
#     Feature selection using Mutual Information
#     """
#     # feature selection
#     selector = SelectKBest(score_func=mutual_info_classif, k=k)
    
#     X_train_selected = selector.fit_transform(X_train, y_train)
#     X_test_selected = selector.transform(X_test)
    
#     return X_train_selected, X_test_selected, selector


# def select_features_rfe(model, X_train, y_train, X_test, k):
#     """
#     Feature selection using Recursive Feature Elimination (RFE) with Random Forest.
#     """
#     # feature selection
#     selector = RFE(estimator=model, n_features_to_select=k, step=1)
    
#     X_train_selected = selector.fit_transform(X_train, y_train)
#     X_test_selected = selector.transform(X_test)
    
#     return X_train_selected, X_test_selected, selector


# def select_features_sfs(model, X_train, y_train, X_test, k, direction='forward'):
#     """
#     Sequential Feature Selection (forward o backward)
#     """
#     # Sequential Feature Selector from sklearn, which iteratively adds (forward) or removes (backward) features based on model performance until k features are selected.
#     sfs = SequentialFeatureSelector(
#         model,
#         n_features_to_select=k,
#         direction=direction,
#         scoring='f1_macro',
#         cv=3,
#         n_jobs=-1,
#     )
    
#     X_train_selected = sfs.fit_transform(X_train, y_train)
#     X_test_selected = sfs.transform(X_test)
    
#     return X_train_selected, X_test_selected, sfs


# def select_features_embedded(X_train, y_train, X_test, C=0.1, l1_ratio=None):
#     scaler = StandardScaler()
    
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
    
#     model = LogisticRegression(
#         solver='saga',
#         C=C,
#         l1_ratio=l1_ratio,
#         max_iter=5000,
#         random_state=42,
#     )
    
#     selector = SelectFromModel(model)
    
#     X_train_selected = selector.fit_transform(X_train_scaled, y_train)
#     X_test_selected = selector.transform(X_test_scaled)
    
#     return X_train_selected, X_test_selected, selector


# def select_features_rf_importance(X_train, y_train, X_test, k):
#     """
#     Feature selection based on Random Forest importance
#     """
#     # train a Random Forest model to compute feature importances and select the top k features based on importance scores.
#     model = RandomForestClassifier(
#         n_estimators=100,
#         random_state=42,
#         class_weight='balanced'
#     )
    
#     model.fit(X_train, y_train)
#     # get feature importances and select top k features
#     importances = model.feature_importances_
#     indices = np.argsort(importances)[::-1][:k]
#     selector = indices  # store the indices of the selected features
    
#     X_train_selected = X_train.iloc[:, indices]
#     X_test_selected = X_test.iloc[:, indices]
    
#     return X_train_selected, X_test_selected, selector

""" 
selectors.py:
Module for various feature selection techniques, including filter methods (Chi-square, Mutual Information),
wrapper methods (RFE), and embedded methods (Random Forest importance). Each function takes training and test data,
applies the specified feature selection method, and returns the transformed datasets along with the fitted selector object for analysis.
The get_selected_feature_names utility function extracts the names of the selected features from any selector object,
facilitating post-hoc analysis of feature importance across different methods and folds.
"""


import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def select_features_chi2(X_train, y_train, X_test, k):
    """
    Filter method: Chi-square test of independence between each feature and the target.
    Requires non-negative values — applies MinMaxScaler unconditionally to [0,1].

    Note: chi2 measures dependence between categorical-like distributions;
    it does NOT capture non-linear interactions. Works best with sparse/count features.
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
    Captures non-linear dependencies. No scaling required.
    Uses random_state for reproducibility (MI uses k-NN internally).
    """
    selector = SelectKBest(score_func=lambda X, y: mutual_info_classif(X, y, random_state=42), k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    return X_train_selected, X_test_selected, selector


def select_features_rfe(model, X_train, y_train, X_test, k):
    heavy_types = (RandomForestClassifier, LogisticRegression)
    try:
        heavy_types = (*heavy_types, XGBClassifier)
    except ImportError:
        pass

    estimator = DecisionTreeClassifier(random_state=42) if isinstance(model, heavy_types) else model

    selector = RFE(estimator=estimator, n_features_to_select=k, step=1)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    return X_train_selected, X_test_selected, selector

def select_features_rf_importance(X_train, y_train, X_test, k):
    """
    Embedded method: Random Forest feature importance (mean decrease in impurity).
    Trains a dedicated RF to rank features; top-k indices are kept.

    Important: this trains a *separate* RF from the classifier — it's used purely
    for feature ranking. class_weight='balanced' compensates for the heavy class
    imbalance in this dataset (OURS dominates ~74% of samples).

    Returns selected columns as DataFrame to preserve feature names downstream.
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("select_features_rf_importance requires a DataFrame input to track feature names.")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    importances = model.feature_importances_
    top_k_indices = np.argsort(importances)[::-1][:k]

    # store both indices and feature names for analysis
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
    Utility to extract selected feature names from any selector object.
    Needed for post-hoc analysis of which features were chosen per fold.

    Parameters
    ----------
    selector : fitted selector object (SelectKBest, RFE, or rf_importance dict)
    feature_names : list or Index of original feature names
    method : str, one of 'chi2', 'mutual_info', 'rfe', 'rf_importance'

    Returns
    -------
    list of selected feature names
    """
    if method in ('chi2', 'mutual_info'):
        # SelectKBest: get_support() returns boolean mask
        return list(np.array(feature_names)[selector.get_support()])

    elif method == 'rfe':
        # RFE: support_ is boolean mask
        return list(np.array(feature_names)[selector.support_])

    elif method == 'rf_importance':
        # our custom dict
        return selector['feature_names']
    
    elif method == 'pca':
        # PCA doesn't select features, it creates components. Return component names.
        return feature_names

    else:
        raise ValueError(f"Unknown method: {method}")