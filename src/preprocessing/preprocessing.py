""" 
preprocessing.py:
Module for data loading and preprocessing in classification-based merge conflict resolution.
Includes functions for loading data, filtering projects based on sample size, creating label columns, and removing
leakage features. This ensures that the dataset is clean and ready for feature selection and model training, 
while also preventing data leakage that could bias the evaluation results.
"""

import pandas as pd

def load_data(file_path):
    """
    Load the dataset from a CSV file.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
def preprocess_data(data):
    """
    Preprocess the dataset by filtering projects with more than 1000 samples and creating a new label column based on conflict resolution results.
    """
    # filter projects with more than 1000 samples
    counts = data['project_id'].value_counts()
    valid_projects = counts[counts > 1000].index
    data_filtered = data[data['project_id'].isin(valid_projects)].copy()

    # create a new label column based on conflict resolution results
    merge_set = [
        "CHUNK_SEMICANONICAL_OURSBASETHEIRS",
        "CHUNK_SEMICANONICAL_BASETHEIRS",
        "CHUNK_SEMICANONICAL_EMPTY",
        "CHUNK_SEMICANONICAL_OURSBASE",
    ]

    data_filtered['label'] = data_filtered['conflictResolutionResult'].apply(
        lambda x: "CHUNK_SEMICANONICAL_OTHERS" if x in merge_set else x
    )
    
    merge_counts = data_filtered.groupby('project_name').agg(
        merges=('merge_id', 'nunique'),
        chunks=('chunk_id', 'count')
    )
    print(merge_counts.sort_values('chunks', ascending=False))

    print("After filtering, number of projects:", data_filtered['project_id'].nunique())
    print("Label distribution:\n", data_filtered['label'].value_counts())
    
    return data_filtered

def remove_leakage_features(data):
    """
    Remove features that could cause data leakage. This function identifies columns that are not needed for training and drops them from both the training and testing sets.
    """
    # drop columns that are not needed for training and that could cause data leakage
    cols_to_remove = [
        'conflictResolutionResult', 'label', 'merge_id', 'project_id',
        'project_name', 'remote_url', 'merge_time', 'file_report_id',
        'file_path', 'chunk_id', 'developersIntersection'
    ]

    data_filtered = data.drop(columns=cols_to_remove, errors='ignore')
        
    return data_filtered