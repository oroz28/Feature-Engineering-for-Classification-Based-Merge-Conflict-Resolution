from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def get_model(model_type):
    if model_type == 'rf':
        return RandomForestClassifier(n_estimators=400, criterion='entropy', max_features='sqrt', min_samples_leaf=1, random_state=42)
    elif model_type == 'lr':
        return LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    elif model_type == 'svm':
        return SVC(kernel='rbf', class_weight='balanced', random_state=42)
    elif model_type == 'knn':
        return KNeighborsClassifier(n_neighbors=5)
    elif model_type == 'dt':
        return DecisionTreeClassifier(class_weight='balanced', random_state=42)
    else:
        raise ValueError("Invalid model type. Please choose from 'rf', 'lr', 'svm', 'knn', or 'dt'.")
    
    