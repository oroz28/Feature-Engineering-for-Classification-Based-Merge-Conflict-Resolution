import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from src.models.models import get_model
from sklearn.metrics import f1_score, accuracy_score, classification_report
from src.preprocessing.preprocessing import remove_leakage_features
from src.features.pipeline import feature_selection_pipeline
from sklearn.preprocessing import StandardScaler

def evaluate(y_test, y_pred):
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average='macro'),
        "f1_weighted": f1_score(y_test, y_pred, average='weighted'),
        "f1_micro": f1_score(y_test, y_pred, average='micro'),
        "classification_report": classification_report(y_test, y_pred, zero_division=0)
    }
    
def evaluate_project_cv(data, model_name='rf', method='baseline', k=None, n_splits=5):
    X = remove_leakage_features(data)
    y = data['label']
    groups = data['merge_id']
    
    print(f"Feature matrix shape: {X.shape}")
    print(X.shape)
    print(X.columns[:10])
    print(y.value_counts().head())
    print(groups.nunique())
    
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    project_name = data['project_name'].iloc[0]
    
    acc_list, f1_weighted_list, f1_macro_list, reports = [], [], [], []
    
    for train_idx, test_idx in sgkf.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # print(f"\nProject: {project_name} | Fold: {len(acc_list)+1}/{n_splits} | Train samples: {len(X_train)} | Test samples: {len(X_test)}")
        
        model = get_model(model_name)
        
        if method != 'baseline':
            X_train, X_test, selector = feature_selection_pipeline(
                X_train, y_train, X_test,
                method=method,
                model=model,
                k=k
            )
        if model_name in ['lr', 'svm', 'knn']:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = evaluate(y_test, y_pred)
        acc_list.append(metrics['accuracy'])
        f1_weighted_list.append(metrics['f1_weighted'])
        f1_macro_list.append(metrics['f1_macro'])
        reports.append(metrics['classification_report'])
    
    return {
        "project_name": project_name,
        "accuracy_mean": np.mean(acc_list),
        "accuracy_std": np.std(acc_list),
        "f1_weighted_mean": np.mean(f1_weighted_list),
        "f1_weighted_std": np.std(f1_weighted_list),
        "f1_macro_mean": np.mean(f1_macro_list),
        "f1_macro_std": np.std(f1_macro_list),
        "classification_report": reports,
        "n_samples": len(X)
    }

def run_experiment_5fold(data, model_name='rf', method='baseline', k=None, n_splits=5):
    results = []
    
    groups1 = list(data.groupby("project_id").groups.keys())
    print("Projects to evaluate:", groups1)

    for project_id, df_proj in data.groupby('project_id'):
        res = evaluate_project_cv(data=df_proj, model_name=model_name, method=method, k=k, n_splits=n_splits)
        results.append(res)

    results_df = pd.DataFrame(results).sort_values(by='accuracy_mean', ascending=False)

    print("\n=== RESUTLS PER PROJECT ===")
    print(results_df.drop(columns=['classification_report', 'n_samples']))
    print("\n=== GLOBAL MEAN ===")
    print("Accuracy mean:", results_df['accuracy_mean'].mean(), "±", results_df['accuracy_std'].mean())
    print("F1 weighted mean:", results_df['f1_weighted_mean'].mean(), "±", results_df['f1_weighted_std'].mean())
    print("F1 macro mean:", results_df['f1_macro_mean'].mean(), "±", results_df['f1_macro_std'].mean())
    print("Classification report for each project:")
    for _, row in results_df.iterrows():
        print(f"  {row['project_name']}:")
        for i, report in enumerate(row['classification_report']):
            print(f"    Fold {i+1}:\n{report}")
    print("Total samples:", results_df['n_samples'].sum())

    return results_df