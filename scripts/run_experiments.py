import pandas as pd
import argparse
from pathlib import Path

from src.preprocessing.preprocessing import load_data, preprocess_data
from src.evalutaion.evalutaion import run_experiment_5fold

# config
DATA_PATH = "data/raw/dataset_chunks_RQ1(in).csv"

def parse_args():
    parser = argparse.ArgumentParser(description="Run ML experiments")
    
    parser.add_argument('--model', type=str, required=True,
                        choices=['rf', 'lr', 'svm', 'knn', 'dt'],
                        help='Model to use')

    parser.add_argument('--method', type=str, required=True,
                        choices=['baseline', 'chi2', 'mutual_info', 'rfe', 'sfs', 'sbs', 'lasso', 'rf_importance'],
                        help='Feature selection method')

    parser.add_argument('--k', type=int, default=None,
                        help='Number of features to select')

    return parser.parse_args()


def run():
    args = parse_args()
    
    model = args.model
    feature_method = args.method
    k = args.k if args.k else None
    
    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Preprocessing data...")
    df = preprocess_data(df)
    
    print("Saving preprocessed data to data/processed/preprocessed_merge_conflict_data.csv...")
    
    processed_path = Path("data/processed/preprocessed_merge_conflict_data.csv")
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"Preprocessed data saved to {processed_path}")
    
    print(f"\nRunning experiment: model={model}, method={feature_method}, k={k if feature_method != 'lasso' else 'N/A'}\n")

    run_experiment_5fold(df, model_name=model, method=feature_method, k=k, n_splits=5)
    
if __name__ == "__main__":
    run()