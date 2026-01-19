from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

def get_models():
    """
    Returns a dictionary of baseline models wrapped in pipelines with scaling.
    Scaling is crucial for SVM, KNN, LR.
    """
    models = {
        "SVM": make_pipeline(StandardScaler(), SVC(random_state=42)),
        "Random Forest": make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42)), # Scaling not strictly needed but harmless
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(random_state=42)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier()),
        "Naive Bayes": make_pipeline(StandardScaler(), GaussianNB())
    }
    return models

def evaluate_model(model, X, y, cv=5):
    """
    Evaluates a model using Stratified K-Fold Cross-Validation.
    Returns mean metrics.
    """
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'kappa': make_scorer(cohen_kappa_score)
    }

    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    scores = cross_validate(model, X, y, cv=cv_strategy, scoring=scoring)

    results = {
        'Accuracy': np.mean(scores['test_accuracy']),
        'Precision': np.mean(scores['test_precision']),
        'Recall': np.mean(scores['test_recall']),
        'F1-Score': np.mean(scores['test_f1']),
        'Kappa': np.mean(scores['test_kappa'])
    }

    return results
