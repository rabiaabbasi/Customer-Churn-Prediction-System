
import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from data_preprocessing import load_data, prepare_data, build_preprocessor


DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

os.makedirs("models", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)


def train_models():

    print("Loading dataset...")

    df = load_data(DATA_PATH)

    X, y = prepare_data(df)

    print("Dataset shape:", X.shape)

    # Split data before fitting preprocessing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    preprocessor = build_preprocessor(X_train)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
    }

    results = []
    trained_models = {}

    for name, algorithm in models.items():

        print(f"\nTraining {name}...")

        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("model", algorithm)
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, predictions, zero_division=0
            ),
            "ROC AUC": roc_auc_score(
                y_test, probabilities
            )
        }

        results.append(metrics)
        trained_models[name] = pipeline

        print(metrics)

    # Compare all models
    comparison = pd.DataFrame(results)

    comparison = comparison.sort_values(
        by="ROC AUC",
        ascending=False
    )

    comparison.to_csv(
        "reports/model_comparison.csv",
        index=False
    )

    # Select best model
    best_name = comparison.iloc[0]["Model"]
    best_model = trained_models[best_name]

    # Save model
    joblib.dump(best_model, "models/best_model.joblib")

    # Save model metadata
    metadata = {
        "best_model": best_name,
        "selection_metric": "ROC AUC",
        "test_size": 0.20,
        "random_state": 42,
        "features": X.columns.tolist()
    }

    with open("models/model_metadata.json", "w") as file:
        json.dump(metadata, file, indent=4)

    print("\nModel comparison:")
    print(comparison.to_string(index=False))

    print("\nBest model:", best_name)
    print("Model saved successfully!")

    return best_model, X_test, y_test


if __name__ == "__main__":
    train_models()
