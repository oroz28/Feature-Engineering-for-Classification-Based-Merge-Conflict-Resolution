
import numpy as np

from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector


def select_features_chi2(X_train, y_train, X_test, k):
    """
    Selects the k best features using Chi-square.
    """
    if (X_train < 0).sum().sum() > 0 or (X_test < 0).sum().sum() > 0:
        print("Warning: Negative values found, applying MinMaxScaler.")
        # scale to [0,1] (necessary for chi2)
        scaler = MinMaxScaler()
    
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
    # feature selection
    selector = SelectKBest(score_func=chi2, k=k)
    
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)    
    # selected_features = X_train.columns[selector.get_support()]
    
    return X_train_selected, X_test_selected, selector


def select_features_mutual_info(X_train, y_train, X_test, k):
    """
    Feature selection using Mutual Information
    """
    # feature selection
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    return X_train_selected, X_test_selected, selector


def select_features_rfe(model, X_train, y_train, X_test, k):
    """
    Feature selection using Recursive Feature Elimination (RFE) with Random Forest.
    """
    # feature selection
    selector = RFE(estimator=model, n_features_to_select=k, step=1)
    
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    return X_train_selected, X_test_selected, selector


def select_features_sfs(model, X_train, y_train, X_test, k, direction='forward'):
    """
    Sequential Feature Selection (forward o backward)
    """
    # Sequential Feature Selector from sklearn, which iteratively adds (forward) or removes (backward) features based on model performance until k features are selected.
    sfs = SequentialFeatureSelector(
        model,
        n_features_to_select=k,
        direction=direction,
        scoring='f1_macro',
        cv=3,
        n_jobs=-1
    )
    
    X_train_selected = sfs.fit_transform(X_train, y_train)
    X_test_selected = sfs.transform(X_test)
    
    return X_train_selected, X_test_selected, sfs


def select_features_lasso(X_train, y_train, X_test):
    """
    Feature selection using LASSO (Logistic Regression L1)
    """
    # scale features (necessary for LASSO)
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # LASSO model for feature selection. Logistic Regression with L1 penalty is used for feature selection.
    lasso = LogisticRegression(
        penalty='l1',
        solver='saga',
        max_iter=1000,
        class_weight='balanced'
    )
    
    # feature selection using SelectFromModel, which selects features based on the importance weights from the LASSO model.
    selector = SelectFromModel(lasso)
    
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    return X_train_selected, X_test_selected, selector


def select_features_rf_importance(X_train, y_train, X_test, k):
    """
    Feature selection based on Random Forest importance
    """
    # train a Random Forest model to compute feature importances and select the top k features based on importance scores.
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    # get feature importances and select top k features
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:k]
    selector = indices  # store the indices of the selected features
    
    X_train_selected = X_train.iloc[:, indices]
    X_test_selected = X_test.iloc[:, indices]
    
    return X_train_selected, X_test_selected, selector



