import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


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
    data = data[data['project_id'].isin(valid_projects)]

    # create a new label column based on conflict resolution results
    merge_set = [
    "CHUNK_SEMICANONICAL_OURSBASETHEIRS",
    "CHUNK_SEMICANONICAL_BASETHEIRS",
    "CHUNK_SEMICANONICAL_EMPTY",
    "CHUNK_SEMICANONICAL_OURSBASE",
    "CHUNK_SEMICANONICAL_OURSTHEIRS"
    ]
    
    data['label'] = data['conflictResolutionResult'].apply(
        lambda x: x if x in merge_set else "CHUNK_SEMICANONICAL_OTHERS"
    )
    
    print(f"Remaining projects: {data['project_id'].nunique()}")

    return data

def train_test_split_by_merge(data, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets. The split is done based on 'merge_id' to ensure that all samples from the same merge are in the same set, 
    preventing data leakage. The function also drops columns that are not needed for training.
    """
    # separate features and labels
    X = data.drop(columns=['conflictResolutionResult', 'label'])
    y = data['label']
    groups = data['merge_id']

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train = X.iloc[train_idx].copy()
    X_test  = X.iloc[test_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_test  = y.iloc[test_idx].copy()

    # ensure no merge_id overlap between train and test sets
    assert len(set(X_train['merge_id']).intersection(set(X_test['merge_id']))) == 0

    # print(f"Train set size: {X_train.shape} samples")
    # print(f"Test set size: {X_test.shape} samples")
    # print(f"y_train class distribution:\n{y_train.value_counts()}")
    # print(f"y_test class distribution:\n{y_test.value_counts()}")
    
    return X_train, X_test, y_train, y_test

def remove_leakage_features(X_train, X_test):
    """
    Remove features that could cause data leakage. This function identifies columns that are not needed for training and drops them from both the training and testing sets.
    """
    # drop columns that are not needed for training and that could cause data leakage
    cols_to_remove = [
        'project_id', 'project_name', 'remote_url',
        'merge_id', 'merge_time',
        'file_report_id', 'file_path',
        'chunk_id', 'developersIntersection',
        'conflictResolutionResult'
    ]

    X_train = X_train.drop(columns=cols_to_remove, errors='ignore')
    X_test  = X_test.drop(columns=cols_to_remove, errors='ignore')
    
    # keep only numeric features (after dropping non-numeric ones)
    X_train = X_train.select_dtypes(include=['int64', 'float64'])
    X_test  = X_test.select_dtypes(include=['int64', 'float64'])
    
    return X_train, X_test