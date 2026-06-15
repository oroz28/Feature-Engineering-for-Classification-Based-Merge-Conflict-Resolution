import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.models.models import get_model
from src.preprocessing.preprocessing import remove_leakage_features
from src.features.pipeline import feature_selection_pipeline
from src.features.selectors import get_selected_feature_names

MODELS_REQUIRING_SCALING = {'lr', 'svm', 'knn'}
MODELS_WITH_PROBA = {'rf', 'lr', 'dt', 'xgb'}


def evaluate(y_test, y_pred, y_proba=None):
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average='macro', zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average='weighted', zero_division=0),
        "f1_micro": f1_score(y_test, y_pred, average='micro', zero_division=0),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "auc_ovr": None,
    }
    if y_proba is not None:
        try:
            # roc_auc_score with ovr requires classes present in y_test to match proba columns
            metrics["auc_ovr"] = roc_auc_score(
                y_test, y_proba,
                multi_class='ovr',
                average='weighted',
                labels=np.unique(y_test)
            )
        except Exception:
            metrics["auc_ovr"] = None
    return metrics


def jaccard_stability(selected_sets: list) -> float:
    """
    Mean Jaccard index across all pairs of CV folds.
    Measures how consistently the same features are selected across splits.
    Score in [0, 1]: 1 = identical sets every fold, 0 = completely disjoint.
    """
    if len(selected_sets) < 2:
        return 1.0
    scores = []
    for a, b in combinations(selected_sets, 2):
        sa, sb = set(a), set(b)
        union = len(sa | sb)
        scores.append(len(sa & sb) / union if union > 0 else 1.0)
    return float(np.mean(scores))


def evaluate_project_cv(data, model_name='rf', method='baseline', k=None, n_splits=5, language='js_initial'):
    """
    5-fold StratifiedGroupKFold CV for a single project.
    Groups by merge_id to prevent split-related data leakage (cf. paper RQ1).
    """
    X = remove_leakage_features(data, language=language)
    feature_names = X.columns.tolist()
    y = data['label']
    groups = data['merge_id']

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=42)
    project_name = data['project_name'].iloc[0]

    acc_list, f1_weighted_list, f1_macro_list, auc_list, reports = [], [], [], [], []
    selected_features_per_fold = []

    if model_name == 'xgb':
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index)

    for train_idx, test_idx in sgkf.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = get_model(model_name)

        if method != 'baseline':
            X_train, X_test, selector = feature_selection_pipeline(
                X_train, y_train, X_test,
                method=method,
                model=model,
                k=k
            )
            fold_features = get_selected_feature_names(
                selector, feature_names, method)
            selected_features_per_fold.append(fold_features)

        if model_name in MODELS_REQUIRING_SCALING:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_proba = None
        if model_name in MODELS_WITH_PROBA:
            try:
                y_proba = model.predict_proba(X_test)
            except Exception:
                pass

        metrics = evaluate(y_test, y_pred, y_proba)
        acc_list.append(metrics['accuracy'])
        f1_weighted_list.append(metrics['f1_weighted'])
        f1_macro_list.append(metrics['f1_macro'])
        reports.append(metrics['classification_report'])
        if metrics['auc_ovr'] is not None:
            auc_list.append(metrics['auc_ovr'])

    jaccard = jaccard_stability(
        selected_features_per_fold) if selected_features_per_fold else None

    return {
        "project_name": project_name,
        "accuracy_mean": np.mean(acc_list),
        "accuracy_std": np.std(acc_list),
        "f1_weighted_mean": np.mean(f1_weighted_list),
        "f1_weighted_std": np.std(f1_weighted_list),
        # "f1_macro_mean": np.mean(f1_macro_list),
        # "f1_macro_std": np.std(f1_macro_list),
        "auc_mean": np.mean(auc_list) if auc_list else None,
        "auc_std": np.std(auc_list) if auc_list else None,
        "jaccard_stability": jaccard,
        "classification_report": reports,
        "n_samples": len(X),
        "selected_features_per_fold": selected_features_per_fold,
    }


def run_experiment_5fold(data, model_name='rf', method='baseline', k=None, n_splits=5, language='js_initial'):
    """
    Runs per-project CV and aggregates results.
    Returns (results_df, freq_df).
    """
    results = []
    feature_selection_counts = {}

    groups1 = list(data.groupby("project_id").groups.keys())
    print("Projects to evaluate:", groups1)

    for project_id, df_proj in data.groupby('project_id'):
        res = evaluate_project_cv(
            data=df_proj, model_name=model_name,
            method=method, k=k, n_splits=n_splits,
            language=language
        )
        results.append(res)

        for fold_features in res['selected_features_per_fold']:
            for feat in fold_features:
                feature_selection_counts[feat] = feature_selection_counts.get(
                    feat, 0) + 1

    results_df = pd.DataFrame(results).sort_values(
        by='accuracy_mean', ascending=False)

    print("\n=== RESULTS PER PROJECT ===")
    print(results_df.drop(
        columns=['classification_report', 'n_samples', 'selected_features_per_fold'], errors='ignore'
    ))
    print("\n=== GLOBAL MEAN ===")
    print(f"Accuracy mean:    {results_df['accuracy_mean'].mean():.4f} ± {results_df['accuracy_std'].mean():.4f}")
    print(f"F1 weighted mean: {results_df['f1_weighted_mean'].mean():.4f} ± {results_df['f1_weighted_std'].mean():.4f}")
    # print(f"F1 macro mean:    {results_df['f1_macro_mean'].mean():.4f} ± {results_df['f1_macro_std'].mean():.4f}")
    if results_df['auc_mean'].notna().any():
        print(f"AUC (OvR) mean:   {results_df['auc_mean'].mean():.4f} ± {results_df['auc_std'].mean():.4f}")
    if results_df['jaccard_stability'].notna().any():
        print(f"Jaccard stability:{results_df['jaccard_stability'].mean():.4f} (mean across projects)")
    print(f"Total samples:    {results_df['n_samples'].sum()}")

    freq_df = pd.DataFrame()
    if feature_selection_counts:
        freq_df = (
            pd.Series(feature_selection_counts, name='selection_count')
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={'index': 'feature'})
        )
        total_folds = len(results_df) * n_splits
        freq_df['selection_rate'] = freq_df['selection_count'] / total_folds
        print(
            f"\n=== FEATURE SELECTION FREQUENCY (top 20 / {total_folds} total folds) ===")
        print(freq_df.head(20).to_string(index=False))

    return results_df, freq_df


def run_experiment_cross_project(data, model_name='rf', method='baseline', 
                                  k=None, n_splits=5, language='all'):
    """
    Cross-project generalization: StratifiedGroupKFold with groups=project_id.
    In each fold, train on a subset of projects, test on unseen projects.
    Mirrors the RQ3 cross-project setting from the paper.
    """
    results = []
    feature_selection_counts = {}

    X = remove_leakage_features(data, language=language)
    feature_names = X.columns.tolist()
    y = data['label']
    groups = data['project_id']  # <-- clave: agrupamos por proyecto, no por merge

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    if model_name == 'xgb':
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index)

    acc_list, f1_weighted_list, auc_list = [], [], []
    selected_features_per_fold = []

    for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        train_projects = data['project_name'].iloc[train_idx].unique()
        test_projects  = data['project_name'].iloc[test_idx].unique()
        print(f"  Fold {fold_idx+1}: train={len(train_projects)} projects, "
              f"test={len(test_projects)} projects ({list(test_projects)})")

        model = get_model(model_name)

        if method != 'baseline':
            X_train, X_test, selector = feature_selection_pipeline(
                X_train, y_train, X_test, method=method, model=model, k=k
            )
            fold_features = get_selected_feature_names(selector, feature_names, method)
            selected_features_per_fold.append(fold_features)

        if model_name in MODELS_REQUIRING_SCALING:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_proba = None
        if model_name in MODELS_WITH_PROBA:
            try:
                y_proba = model.predict_proba(X_test)
            except Exception:
                pass

        metrics = evaluate(y_test, y_pred, y_proba)
        acc_list.append(metrics['accuracy'])
        f1_weighted_list.append(metrics['f1_weighted'])
        if metrics['auc_ovr'] is not None:
            auc_list.append(metrics['auc_ovr'])

    for fold_features in selected_features_per_fold:
        for feat in fold_features:
            feature_selection_counts[feat] = feature_selection_counts.get(feat, 0) + 1

    result = {
        "accuracy_mean":    np.mean(acc_list),
        "accuracy_std":     np.std(acc_list),
        "f1_weighted_mean": np.mean(f1_weighted_list),
        "f1_weighted_std":  np.std(f1_weighted_list),
        "auc_mean": np.mean(auc_list) if auc_list else None,
        "auc_std":  np.std(auc_list)  if auc_list else None,
        "jaccard_stability": jaccard_stability(selected_features_per_fold) 
                             if selected_features_per_fold else None,
        "n_samples": len(X),
        "selected_features_per_fold": selected_features_per_fold,
    }

    freq_df = pd.DataFrame()
    if feature_selection_counts:
        freq_df = (
            pd.Series(feature_selection_counts, name='selection_count')
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={'index': 'feature'})
        )
        total_folds = n_splits
        freq_df['selection_rate'] = freq_df['selection_count'] / total_folds
    
    result_df = pd.DataFrame([result])

    return result_df, freq_df

def run_experiment_cross_project_by_language(
        data, model_name='rf', method='baseline', k=None, n_splits=5, language='all'):
    """
    For each language, performs cross-project 5-fold CV.
    If language='all', uses all projects together.
    Replicates Table IV (cross-project) from the paper.
    """
    results = []
    all_freqs = []

    languages = normalize_languages(language, data)
    print(f"Languages found: {sorted(languages)}")

    for lang in sorted(languages):
        df_lang = data[data['programming_language'] == lang]
        n_projects = df_lang['project_id'].nunique()

        if n_projects < n_splits:
            print(f"  Skipping {lang}: only {n_projects} projects "
                  f"(need >= {n_splits})")
            continue

        print(f"\n--- Language: {lang} | Projects: {n_projects} | "
              f"Chunks: {len(df_lang)} ---")

        res_df, freq_df = run_experiment_cross_project(
            df_lang, model_name=model_name, method=method,
            k=k, n_splits=n_splits,
        )
        res_df['language'] = lang
        res_df['n_projects'] = n_projects
        results.append(res_df)
        
        if not freq_df.empty:
            freq_df['language'] = lang
            all_freqs.append(freq_df)
        

    results_df = pd.concat(results, ignore_index=True).sort_values('language')
    
    freqs_df = pd.DataFrame()
    if all_freqs:
        freqs_df = pd.concat(all_freqs, ignore_index=True)

    print("\n=== CROSS-PROJECT RESULTS BY LANGUAGE ===")
    cols = ['language', 'n_projects', 'accuracy_mean', 'accuracy_std',
            'f1_weighted_mean', 'auc_mean']
    print(results_df[[c for c in cols if c in results_df.columns]].to_string(index=False))
    print(f"\nMean accuracy across languages: "
          f"{results_df['accuracy_mean'].mean():.4f} ± "
          f"{results_df['accuracy_std'].mean():.4f}")

    return results_df, freqs_df

def normalize_languages(language, data):
    if language == "all":
        return sorted(data['programming_language'].unique())
    if isinstance(language, str):
        return [language]
    return language