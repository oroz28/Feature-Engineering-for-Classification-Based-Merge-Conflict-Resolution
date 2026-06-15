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
from src.evalutaion.evalutaion import run_experiment_5fold, run_experiment_cross_project_by_language, run_experiment_cross_project

DATA_PATH_JS = "data/raw/dataset_chunks_RQ1(in).csv"
DATA_PATH_ALL = "data/raw/All_Languages_Combined_Unique.csv"

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
    
    parser.add_argument('--language', type=str, default='js_initial',
                        help='Programming languages to include, "all" for all languages')
    
    parser.add_argument('--eval_mode', type=str, default='intra_project',
                    choices=['intra_project', 'cross_project', 'global'],
                    help='intra_project: per-project 5-fold CV (default). '
                         'cross_project: train on subset of projects, test on unseen.')

    return parser.parse_args()


def run():
    args = parse_args()

    print("Loading data...")
    if args.language == 'js_initial':
        print("Loading js_initial dataset...")
        df = load_data(DATA_PATH_JS)
    else:        
        print("Loading all languages dataset...")
        df = load_data(DATA_PATH_ALL)

    print("Preprocessing data...")
    df = preprocess_data(df)

    processed_path = Path("data/processed/preprocessed_merge_conflict_data.csv") if args.language == 'js_initial' else Path("data/processed/preprocessed_merge_conflict_data_all.csv")
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)

    label = f"{args.model}_{args.method}" + (f"_k{args.k}" if args.k else "")
    print(f"\nRunning experiment: {label}\n")

    if args.eval_mode == 'cross_project':
        results_dir = Path("results") / args.language
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_df, freq_df = run_experiment_cross_project_by_language(
            df,
            model_name=args.model,
            method=args.method,
            k=args.k,
            n_splits=5,
            language=args.language
        )
        
    elif args.eval_mode == 'global':
        results_dir = Path("results") / args.model
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_df, freq_df = run_experiment_cross_project(
            df,
            model_name=args.model,
            method=args.method,
            k=args.k,
            n_splits=5
        )
    else:
        results_dir = Path("results") / "Java" / args.model
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_df, freq_df = run_experiment_5fold(
            df,
            model_name=args.model,
            method=args.method,
            k=args.k,
            n_splits=5,
            language=args.language
        )

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