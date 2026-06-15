""" 
run_experiments.py:
Script to run classification experiments for merge conflict resolution with various
feature selection/engineering methods and models. Results are saved as CSVs for
subsequent analysis with results_analysis.py.
"""

import pandas as pd
import argparse
from pathlib import Path

from src.preprocessing.preprocessing import load_data, preprocess_data
from src.evalutaion.evalutaion import run_experiment_5fold

DATA_PATH = "data/raw/dataset_chunks_RQ1(in).csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Run ML experiments")

    parser.add_argument('--model', type=str, required=True,
                        choices=['rf', 'lr', 'svm', 'knn', 'dt', 'xgb'],
                        help='Model to use')

    parser.add_argument('--method', type=str, required=True,
                        choices=['baseline', 'chi2', 'mutual_info', 'rfe', 'rf_importance', 'pca', 'poly'],
                        help='Feature selection/engineering method')

    parser.add_argument('--k', type=int, default=None,
                        help='Number of features/components to select (not needed for baseline)')

    parser.add_argument('--save_freq', action='store_true',
                        help='Save feature selection frequency CSV alongside results')

    return parser.parse_args()


def run():
    args = parse_args()

    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Preprocessing data...")
    df = preprocess_data(df)

    processed_path = Path("data/processed/preprocessed_merge_conflict_data.csv")
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)

    label = f"{args.model}_{args.method}" + (f"_k{args.k}" if args.k else "")
    print(f"\nRunning experiment: {label}\n")

    results_df, freq_df = run_experiment_5fold(
        df,
        model_name=args.model,
        method=args.method,
        k=args.k,
        n_splits=5
    )

    results_dir = Path("results") / args.model
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = results_dir / f"{label}_metrics.csv"
    results_df.drop(
        columns=['classification_report', 'selected_features_per_fold'], errors='ignore'
    ).to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to {metrics_path}")

    if args.save_freq and not freq_df.empty:
        freq_path = results_dir / f"{label}_feature_freq.csv"
        freq_df.to_csv(freq_path, index=False)
        print(f"Feature frequencies saved to {freq_path}")


if __name__ == "__main__":
    run()