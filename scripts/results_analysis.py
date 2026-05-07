import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

RESULTS_DIR = Path("results")
PROCESSED_DATA = Path("data/processed/preprocessed_merge_conflict_data.csv")
METHODS = ["chi2", "mutual_info", "rfe", "rf_importance"]
KS = [10, 20, 30, 40]


# 1. Load helpers
def load_metrics(model):
    """
    Aggregates all per-project metrics CSVs for a given model.
    """
    rows = []
    model_dir = RESULTS_DIR / model

    for tag, method, k in [("baseline", "baseline", np.nan), ("pca", "pca", np.nan)]:
        p = model_dir / f"{model}_{tag}_metrics.csv"
        if p.exists():
            df = pd.read_csv(p)
            df['method'] = method
            df['k'] = k
            rows.append(df)

    for method in METHODS:
        for k in KS:
            p = model_dir / f"{model}_{method}_k{k}_metrics.csv"
            if p.exists():
                df = pd.read_csv(p)
                df['method'] = method
                df['k'] = k
                rows.append(df)

    if not rows:
        raise FileNotFoundError(
            f"No metrics CSVs found in {
                RESULTS_DIR / model} for model='{model}'. "
            "Run run_experiments.sh first."
        )
    return pd.concat(rows, ignore_index=True)


def load_feature_frequencies(model):
    """
    Loads feature selection frequency CSVs for all methods and k values.
    Returns a dict of DataFrames keyed by method_k (e.g., 'chi2_k10').
    """
    freqs = {}
    model_dir = RESULTS_DIR / model
    for method in METHODS:
        for k in KS:
            p = model_dir / f"{model}_{method}_k{k}_feature_freq.csv"
            if p.exists():
                freqs[f"{method}_k{k}"] = pd.read_csv(p)
    return freqs


def load_jaccard(model):
    """
    Load Jaccard stability values from metrics CSVs (stored per project).
    """
    rows = []
    model_dir = RESULTS_DIR / model
    for method in METHODS:
        for k in KS:
            p = model_dir / f"{model}_{method}_k{k}_metrics.csv"
            if p.exists():
                df = pd.read_csv(p)
                if 'jaccard_stability' in df.columns:
                    mean_j = df['jaccard_stability'].mean()
                    rows.append({'method': method, 'k': k,
                                'jaccard_mean': mean_j})
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# 2. Performance comparison
def summarise_performance(metrics_df):
    """
    Aggregates mean and std of accuracy, F1 (weighted), and optionally AUC across projects for each method × k.
    Returns a summary DataFrame with columns: method, k, acc_mean, acc_std, f1w_mean, f1w_std, (optional) auc_mean, auc_std.
    """
    agg_cols = {
        'acc_mean': ('accuracy_mean', 'mean'),
        'acc_std': ('accuracy_std', 'mean'),
        'f1w_mean': ('f1_weighted_mean', 'mean'),
        'f1w_std': ('f1_weighted_std', 'mean'),
        # 'f1m_mean': ('f1_macro_mean', 'mean'),
        # 'f1m_std': ('f1_macro_std', 'mean'),
    }
    if 'auc_mean' in metrics_df.columns:
        agg_cols['auc_mean'] = ('auc_mean', 'mean')
        agg_cols['auc_std'] = ('auc_std', 'mean')

    return (
        metrics_df
        .groupby(['method', 'k'], dropna=False)
        .agg(**agg_cols)
        .reset_index()
        .sort_values(['method', 'k'])
    )


def plot_performance_vs_k(summary, model, save):
    """
    Plots accuracy, F1 (weighted), and optionally AUC vs k for each method, with error bars for std. Baseline and PCA are shown as horizontal lines for reference.
    """
    has_auc = 'auc_mean' in summary.columns and summary['auc_mean'].notna(
    ).any()
    ncols = 3 if has_auc else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))

    metric_configs = [
        ('acc_mean', 'acc_std', 'Accuracy'),
        ('f1w_mean', 'f1w_std', 'F1 Weighted'),
    ]
    if has_auc:
        metric_configs.append(('auc_mean', 'auc_std', 'AUC (OvR weighted)'))

    for ax, (mean_col, std_col, title) in zip(axes, metric_configs):
        for method in METHODS:
            sub = summary[summary['method'] == method].dropna(subset=['k'])
            if sub.empty or sub[mean_col].isna().all():
                continue
            ax.errorbar(sub['k'], sub[mean_col], yerr=sub[std_col],
                        marker='o', label=method, capsize=3)

        for tag in ['baseline', 'pca']:
            base = summary[summary['method'] == tag]
            if not base.empty and not base[mean_col].isna().all():
                ax.axhline(base[mean_col].values[0],
                           linestyle='--', label=tag, alpha=0.7)

        ax.set_xlabel('k (features selected)')
        ax.set_ylabel(title)
        ax.set_title(f'{title} vs k — {model.upper()}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = RESULTS_DIR / model / f"{model}_performance_vs_k.png"
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.show()


# 3. Feature selection frequency
def aggregate_frequency_across_k(freqs, method):
    """ 
    Aggregates feature selection frequencies across different k values for a given method.
    Returns a DataFrame with columns: feature, avg_selection_rate, sorted by avg_selection_rate descending. 
    """
    dfs = [v for label, v in freqs.items() if label.startswith(method)]
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs)[['feature', 'selection_rate']]
    return (
        combined.groupby('feature')['selection_rate']
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={'selection_rate': 'avg_selection_rate'})
    )


def plot_feature_frequency(freqs, model, top_n, save):
    """ 
    Plots the top N most frequently selected features for each method, averaged across k values.
    Each subplot corresponds to a method, showing features on the y-axis and average selection rate on the x-axis."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for ax, method in zip(axes, METHODS):
        agg = aggregate_frequency_across_k(freqs, method)
        if agg.empty:
            ax.set_title(f'{method} — no data')
            continue
        top = agg.head(top_n)
        ax.barh(top['feature'][::-1], top['avg_selection_rate'][::-1])
        ax.set_xlabel('Avg selection rate across k values')
        ax.set_title(f'{method} — top {top_n} features')
        ax.set_xlim(0, 1)
        ax.axvline(0.5, linestyle='--', color='red', alpha=0.4, label='50%')
        ax.legend(fontsize=8)

    plt.suptitle(f'Feature Selection Frequency — {model.upper()}', fontsize=14)
    plt.tight_layout()
    if save:
        path = RESULTS_DIR / model / f"{model}_feature_frequency.png"
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.show()


def find_stable_features(freqs, method, threshold=0.8):
    """ 
    Identifies features that are stably selected (avg selection rate >= threshold) across different k values for a given method. 
    Returns a list of stable features. 
    """
    agg = aggregate_frequency_across_k(freqs, method)
    if agg.empty:
        return []
    return agg[agg['avg_selection_rate'] >= threshold]['feature'].tolist()


# 4. Jaccard stability
def plot_jaccard_stability(jaccard_df, model, save):
    """
    Plots a heatmap of mean Jaccard stability values for each method × k combination.
    """
    if jaccard_df.empty:
        print("No Jaccard data available.")
        return

    pivot = jaccard_df.pivot(
        index='method', columns='k', values='jaccard_mean')
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(pivot, annot=True, fmt='.2f',
                cmap='YlGn', vmin=0, vmax=1, ax=ax)
    ax.set_title(f'Jaccard Stability (mean across projects) — {model.upper()}')
    plt.tight_layout()
    if save:
        path = RESULTS_DIR / model / f"{model}_jaccard_stability.png"
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.show()


# 5. Redundancy: Pearson correlation
def _load_feature_matrix():
    """ 
    Loads the preprocessed feature matrix from the processed data CSV, removing any leakage features. Returns a DataFrame with only the features used for selection. 
    """
    from src.preprocessing.preprocessing import remove_leakage_features
    df = pd.read_csv(PROCESSED_DATA)
    return remove_leakage_features(df)


def compute_feature_correlations():
    """ 
    Computes the Pearson correlation matrix for the features in the preprocessed data. 
    Returns a DataFrame where entry (i, j) is the correlation between feature i and feature j. 
    """
    X = _load_feature_matrix()
    return X.corr(method='pearson')


def find_highly_correlated_pairs(corr_matrix, threshold=0.85):
    """
    Finds pairs of features with absolute correlation above the specified threshold.
    Returns a DataFrame with columns: feature_a, feature_b, correlation, sorted by absolute correlation descending
    """
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    pairs = (
        upper.stack()
        .reset_index()
        .rename(columns={'level_0': 'feature_a', 'level_1': 'feature_b', 0: 'correlation'})
    )
    return pairs[pairs['correlation'].abs() >= threshold].sort_values('correlation', ascending=False)


def plot_correlation_heatmap(corr_matrix, model, save):
    """
    Plots a heatmap of the upper triangle of the correlation matrix to visualize feature redundancy.
    """
    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, cmap='coolwarm', center=0,
        vmin=-1, vmax=1, square=True, linewidths=0.3,
        cbar_kws={"shrink": 0.7}, ax=ax, annot=False
    )
    ax.set_title('Feature Correlation Matrix (Pearson)', fontsize=13)
    plt.tight_layout()
    if save:
        path = RESULTS_DIR / model / f"{model}_correlation_heatmap.png"
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.show()


# 6. VIF — multicollinearity analysis (critical for LR)
def compute_vif():
    """
    Variance Inflation Factor for each feature.
    VIF = 1/(1 - R ^ 2) where R ^ 2 is from regressing that feature on all others.
    VIF > 5 is concerning, VIF > 10 indicates severe multicollinearity.
    Particularly relevant for LR where correlated features destabilize coefficients.
    """
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError:
        print("statsmodels not installed. Run: pip install statsmodels")
        return pd.DataFrame()

    from sklearn.preprocessing import StandardScaler

    X = _load_feature_matrix()
    # VIF is scale-sensitive; standardize first
    X_scaled = StandardScaler().fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    vif_data = pd.DataFrame({
        'feature': X_scaled.columns,
        'VIF': [variance_inflation_factor(X_scaled.values, i) for i in range(X_scaled.shape[1])]
    })
    return vif_data.sort_values('VIF', ascending=False).reset_index(drop=True)


def plot_vif(vif_df, model, save, threshold=10.0):
    """
    Plots VIF values as a horizontal bar chart, highlighting features above the threshold.
    """
    if vif_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['red' if v > threshold else 'steelblue' for v in vif_df['VIF']]
    ax.barh(vif_df['feature'][::-1], vif_df['VIF'][::-1], color=colors[::-1])
    ax.axvline(threshold, linestyle='--', color='red',
               alpha=0.6, label=f'VIF={threshold} threshold')
    ax.axvline(5, linestyle='--', color='orange',
               alpha=0.6, label='VIF=5 warning')
    ax.set_xlabel('Variance Inflation Factor (VIF)')
    ax.set_title(f'Feature Multicollinearity (VIF) — {model.upper()}')
    ax.legend()
    plt.tight_layout()
    if save:
        path = RESULTS_DIR / model / f"{model}_vif.png"
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.show()


# 7. PCA variance analysis
def run_pca_variance_analysis(model, save):
    """
    Fits full PCA on the feature matrix and plots cumulative explained variance.
    Identifies how many components are needed for 90/95/99 % variance.
    Motivates whether PCA is a viable alternative to feature selection here.
    """
    from src.features.feature_engineering import pca_variance_analysis

    X = _load_feature_matrix()
    result = pca_variance_analysis(X, scale_first=True)

    print(f"\n--- PCA Variance Analysis ---")
    print(f"  Total features: {result['total_features']}")
    print(f"  Components for 90% variance: {result['n_components_90']}")
    print(f"  Components for 95% variance: {result['n_components_95']}")
    print(f"  Components for 99% variance: {result['n_components_99']}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, len(result['cumulative_variance']) + 1),
            result['cumulative_variance'], marker='o', markersize=3)
    for thresh, n, color in [(0.90, result['n_components_90'], 'orange'),
                             (0.95, result['n_components_95'], 'red'),
                             (0.99, result['n_components_99'], 'darkred')]:
        ax.axhline(thresh, linestyle='--', color=color, alpha=0.6,
                   label=f'{int(thresh*100)}% ({n} components)')
        ax.axvline(n, linestyle=':', color=color, alpha=0.4)

    ax.set_xlabel('Number of principal components')
    ax.set_ylabel('Cumulative explained variance')
    ax.set_title(f'PCA Cumulative Explained Variance — {model.upper()}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        path = RESULTS_DIR / model / f"{model}_pca_variance.png"
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.show()

    return result


# 8. Optimal subset recommendation
def recommend_optimal_subset(summary, freqs, stable_threshold=0.8, corr_threshold=0.85):
    """
    Recommends an optimal feature subset based on performance, stability, and redundancy.
    1. Selects best-performing method × k(excluding baseline and PCA).
    2. Finds stably selected features for that method(selection rate >= stable_threshold).
    3. Computes correlation matrix and removes one feature from each highly correlated pair ( | r | >= corr_threshold).
    Returns a dict with the recommended features and rationale.
    """
    non_base = summary[~summary['method'].isin(['baseline', 'pca'])]
    if non_base.empty:
        return {"error": "No non-baseline results available."}

    best_row = non_base.loc[non_base['acc_mean'].idxmax()]
    best_method = best_row['method']
    best_k = int(best_row['k'])

    stable = find_stable_features(freqs, best_method, stable_threshold)

    try:
        corr = compute_feature_correlations()
        high_corr_pairs = find_highly_correlated_pairs(corr, corr_threshold)
        to_remove = set()
        for _, row in high_corr_pairs.iterrows():
            if row['feature_a'] not in to_remove:
                to_remove.add(row['feature_b'])
        final_features = [f for f in stable if f not in to_remove]
    except Exception as e:
        print(f"Warning: redundancy filter failed ({e}).")
        final_features = stable
        to_remove = set()

    return {
        "best_method": best_method,
        "best_k": best_k,
        "best_accuracy": round(best_row['acc_mean'], 4),
        "stable_features_before_dedup": stable,
        "redundant_features_removed": sorted(to_remove & set(stable)),
        "recommended_features": final_features,
        "n_recommended": len(final_features),
    }

# 9. Statistical significance testing (Wilcoxon signed-rank test)


def run_wilcoxon_tests(model):
    """
    For each candidate method × k, performs Wilcoxon signed-rank test against baseline.
    Prints mean accuracy, p-value, and whether the difference is statistically significant.
    Note: assumes metrics CSVs contain per-project accuracy values(not just means)."""
    from scipy.stats import wilcoxon

    model_dir = RESULTS_DIR / model
    baseline_path = model_dir / f"{model}_baseline_metrics.csv"
    if not baseline_path.exists():
        print(f"  Baseline not found for {model}")
        return

    baseline_acc = pd.read_csv(baseline_path)['accuracy_mean'].values
    baseline_mean = baseline_acc.mean()

    # Build candidates dynamically from all available CSVs
    candidates = []
    for method in METHODS:
        for k in KS:
            p = model_dir / f"{model}_{method}_k{k}_metrics.csv"
            if p.exists():
                candidates.append((method, k))
    # Add pca and baseline variants without k
    for tag in ['pca']:
        p = model_dir / f"{model}_{tag}_metrics.csv"
        if p.exists():
            candidates.append((tag, None))

    print(f"\n--- Wilcoxon Signed-Rank Tests vs Baseline — "
          f"{model.upper()} (n=16 projects) ---")
    print(f"  {'configuration':<25} {'acc_mean':>10} {'vs_baseline':>12} "
          f"{'p-value':>10} {'significant':>12}")
    print("  " + "-" * 72)

    for method, k in candidates:
        if k is not None:
            p_path = model_dir / f"{model}_{method}_k{k}_metrics.csv"
            label = f"{method}_k{k}"
        else:
            p_path = model_dir / f"{model}_{method}_metrics.csv"
            label = method

        candidate_acc = pd.read_csv(p_path)['accuracy_mean'].values

        if np.all(baseline_acc == candidate_acc):
            print(f"  {label:<25} {candidate_acc.mean():>10.4f} "
                  f"{'identical':>12}")
            continue

        try:
            stat, p_val = wilcoxon(baseline_acc, candidate_acc)
            sig = 'YES *' if p_val < 0.05 else 'no'
            delta = candidate_acc.mean() - baseline_mean
            print(f"  {label:<25} {candidate_acc.mean():>10.4f} "
                  f"{delta:>+12.4f} {p_val:>10.4f} {sig:>12}")
        except Exception as e:
            print(f"  {label:<25} ERROR: {e}")


# Main
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='rf')
    p.add_argument('--save_plots', action='store_true')
    p.add_argument('--top_n_features', type=int, default=20)
    p.add_argument('--corr_threshold', type=float, default=0.85)
    p.add_argument('--stable_threshold', type=float, default=0.8)
    p.add_argument('--vif_threshold', type=float, default=10.0)
    return p.parse_args()


def main():
    args = parse_args()
    model = args.model
    model_dir = RESULTS_DIR / model
    model_dir.mkdir(exist_ok=True)

    print(f"=== Results Analysis — model: {model.upper()} ===\n")

    # Performance
    print("Loading metrics...")
    metrics_df = load_metrics(model)
    summary = summarise_performance(metrics_df)

    print("\n--- Performance Summary (global mean across projects) ---")
    print(summary.to_string(index=False))
    summary.to_csv(model_dir / f"{model}_performance_summary.csv", index=False)

    plot_performance_vs_k(summary, model, args.save_plots)

    # Feature frequencies
    print("\nLoading feature frequencies...")
    freqs = load_feature_frequencies(model)

    if freqs:
        plot_feature_frequency(
            freqs, model, args.top_n_features, args.save_plots)

        print(
            f"\n--- Stably selected features (rate >= {args.stable_threshold:.0%}) per method ---")
        for method in METHODS:
            stable = find_stable_features(freqs, method, args.stable_threshold)
            print(f"  {method:15s}: {len(stable):2d} — {stable}")
    else:
        print("No frequency CSVs found. Re-run with --save_freq.")

    # Jaccard stability
    jaccard_df = load_jaccard(model)
    if not jaccard_df.empty:
        print("\n--- Jaccard Stability (mean across projects per method × k) ---")
        print(jaccard_df.to_string(index=False))
        plot_jaccard_stability(jaccard_df, model, args.save_plots)
    else:
        print(
            "\nNo Jaccard data found (jaccard_stability column missing from metrics CSVs).")

    # Redundancy: correlation
    print("\n--- Feature Redundancy (Pearson correlation) ---")
    if PROCESSED_DATA.exists():
        corr = compute_feature_correlations()
        high_corr = find_highly_correlated_pairs(corr, args.corr_threshold)
        print(f"Pairs with |r| >= {args.corr_threshold}: {len(high_corr)}")
        if not high_corr.empty:
            print(high_corr.to_string(index=False))
        plot_correlation_heatmap(corr, model, args.save_plots)
    else:
        print(f"Preprocessed data not found at {PROCESSED_DATA}.")

    # VIF multicollinearity analysis
    print("\n--- VIF Multicollinearity Analysis ---")
    vif_df = compute_vif()
    if not vif_df.empty:
        severe = vif_df[vif_df['VIF'] > args.vif_threshold]
        print(f"Features with VIF > {
              args.vif_threshold} (severe multicollinearity): {len(severe)}")
        print(vif_df.to_string(index=False))
        vif_df.to_csv(model_dir / f"{model}_vif.csv", index=False)
        plot_vif(vif_df, model, args.save_plots, args.vif_threshold)

    # PCA variance analysis
    if PROCESSED_DATA.exists():
        run_pca_variance_analysis(model, args.save_plots)

    # Polynomial expansion analysis
    polynomial_feature_count_analysis()

    # Recommendation
    if freqs and PROCESSED_DATA.exists():
        print("\n--- Optimal Feature Subset Recommendation ---")
        rec = recommend_optimal_subset(
            summary, freqs,
            stable_threshold=args.stable_threshold,
            corr_threshold=args.corr_threshold,
        )
        for key, val in rec.items():
            print(f"  {key}: {val}")

    # Statistical significance testing
    run_wilcoxon_tests(model)


if __name__ == "__main__":
    main()
