""" 
models.py:
Module for defining and retrieving machine learning models used in classification-based merge conflict resolution.
Provides a get_model function that returns an instance of the specified model type with predefined hyperparameters.
This centralizes model definitions and allows for easy updates to model configurations across the entire codebase.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def get_model(model_type):
    if model_type == 'rf':
        return RandomForestClassifier(n_estimators=400, criterion='entropy', max_features=0.3, min_samples_leaf=1, random_state=42, n_jobs=-1)
    elif model_type == 'lr':
        return LogisticRegression(max_iter=10000, class_weight='balanced', random_state=42)
    elif model_type == 'svm':
        return SVC(kernel='rbf', class_weight='balanced', random_state=42)
    elif model_type == 'knn':
        return KNeighborsClassifier(n_neighbors=5)
    elif model_type == 'dt':
        return DecisionTreeClassifier(class_weight='balanced', random_state=42)
    elif model_type == 'xgb':
        return XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.3,
            reg_alpha=0.5,
            reg_lambda=1.0,
            min_child_weight=5,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(
            "Invalid model type. Please choose from 'rf', 'lr', 'svm', 'knn', or 'dt'.")
