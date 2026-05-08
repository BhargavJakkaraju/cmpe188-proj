"""
Best Model Selection and Optimization
Early-Stage Diabetes Risk Prediction System

Reads baseline model comparison results, selects the current best model
(by AUC-ROC), runs hyperparameter optimization, and exports optimized artifacts.
"""

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier


DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RANDOM_STATE = 42


def load_data():
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")
    return X_train, X_test, y_train, y_test


def evaluate_binary_classifier(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "auc_roc": float(auc(fpr, tpr)),
    }


def optimize_xgboost(X_train, y_train):
    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )

    param_distributions = {
        "n_estimators": [150, 250, 350, 450, 550],
        "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1],
        "max_depth": [3, 4, 5, 6],
        "min_child_weight": [1, 3, 5, 7],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "gamma": [0.0, 0.1, 0.2, 0.3],
        "reg_alpha": [0.0, 0.01, 0.1, 1.0],
        "reg_lambda": [1.0, 1.5, 2.0, 3.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=35,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("BEST MODEL SELECTION + OPTIMIZATION")
    print("=" * 70)

    metrics_df = pd.read_csv(RESULTS_DIR / "model_metrics.csv")
    baseline_best = metrics_df.sort_values("auc_roc", ascending=False).iloc[0]
    print(f"Baseline best model (AUC-ROC): {baseline_best['model']}")

    if baseline_best["model"] != "XGBoost":
        print("Current optimizer targets XGBoost only. Re-run training if needed.")
        return

    X_train, X_test, y_train, y_test = load_data()

    baseline_model = joblib.load(RESULTS_DIR / "xgboost.pkl")
    baseline_test_metrics = evaluate_binary_classifier(baseline_model, X_test, y_test)

    print("\nRunning RandomizedSearchCV for XGBoost...")
    search = optimize_xgboost(X_train, y_train)
    best_model = search.best_estimator_
    tuned_test_metrics = evaluate_binary_classifier(best_model, X_test, y_test)

    joblib.dump(best_model, RESULTS_DIR / "best_model_optimized.pkl")

    comparison = pd.DataFrame(
        [
            {"version": "baseline_xgboost", **baseline_test_metrics},
            {"version": "optimized_xgboost", **tuned_test_metrics},
        ]
    )
    comparison.to_csv(RESULTS_DIR / "best_model_optimization_comparison.csv", index=False)

    details = {
        "selected_best_model_from_baseline": "XGBoost",
        "search_method": "RandomizedSearchCV",
        "search_scoring": "roc_auc",
        "cv_folds": 5,
        "n_iter": 35,
        "best_cv_score_auc_roc": float(search.best_score_),
        "best_params": search.best_params_,
        "baseline_test_metrics": baseline_test_metrics,
        "optimized_test_metrics": tuned_test_metrics,
    }
    with open(RESULTS_DIR / "best_model_optimization_details.json", "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2)

    print("\n" + "-" * 70)
    print("BASELINE XGBOOST TEST METRICS")
    print(baseline_test_metrics)
    print("\nOPTIMIZED XGBOOST TEST METRICS")
    print(tuned_test_metrics)
    print("\nBEST PARAMETERS")
    print(search.best_params_)

    print("\nSaved:")
    print(" - results/best_model_optimized.pkl")
    print(" - results/best_model_optimization_comparison.csv")
    print(" - results/best_model_optimization_details.json")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
