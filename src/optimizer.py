import optuna
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import warnings

# Suppress convergence warnings for cleaner output
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def objective(trial, model_name, X, y):
    if model_name == "SVM":
        c = trial.suggest_float("C", 1e-3, 100, log=True)
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"])
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
        clf = SVC(C=c, kernel=kernel, gamma=gamma, random_state=42)

    elif model_name == "Random Forest":
        n_estimators = trial.suggest_int("n_estimators", 10, 200)
        max_depth = trial.suggest_int("max_depth", 2, 30)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 15)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )

    elif model_name == "Logistic Regression":
        c = trial.suggest_float("C", 1e-3, 100, log=True)
        solver = trial.suggest_categorical("solver", ["liblinear", "lbfgs"])
        if solver == "liblinear":
            penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
        else:
            penalty = "l2" # lbfgs supports l2

        clf = LogisticRegression(C=c, solver=solver, penalty=penalty, random_state=42)

    elif model_name == "KNN":
        n_neighbors = trial.suggest_int("n_neighbors", 3, 20)
        weights = trial.suggest_categorical("weights", ["uniform", "distance"])
        metric = trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"])
        clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)

    elif model_name == "Naive Bayes":
        var_smoothing = trial.suggest_float("var_smoothing", 1e-9, 1e-1, log=True)
        clf = GaussianNB(var_smoothing=var_smoothing)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Use pipeline with scaling for all models to ensure valid cross-validation
    pipeline = make_pipeline(StandardScaler(), clf)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1")
    return scores.mean()

def optimize_model(model_name, X, y, n_trials=20):
    """
    Runs Optuna optimization for a specific model.
    Returns the best model pipeline and the study.
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, model_name, X, y), n_trials=n_trials)

    best_params = study.best_params

    # Reconstruct the best model
    if model_name == "SVM":
        best_clf = SVC(**best_params, random_state=42)
    elif model_name == "Random Forest":
        best_clf = RandomForestClassifier(**best_params, random_state=42)
    elif model_name == "Logistic Regression":
        best_clf = LogisticRegression(**best_params, random_state=42)
    elif model_name == "KNN":
        best_clf = KNeighborsClassifier(**best_params)
    elif model_name == "Naive Bayes":
        best_clf = GaussianNB(**best_params)

    best_pipeline = make_pipeline(StandardScaler(), best_clf)

    return best_pipeline, best_params, study.best_value
