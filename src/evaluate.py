
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

from data_preprocessing import load_data, prepare_data
from sklearn.model_selection import train_test_split


def evaluate_model():

    model = joblib.load("models/best_model.joblib")

    df = load_data(
        "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    X, y = prepare_data(df)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["No Churn", "Churn"]
        )
    )

    print("ROC-AUC:", roc_auc_score(y_test, probabilities))

    os.makedirs("reports/figures", exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"]
    )

    plt.title("Customer Churn Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    plt.savefig("reports/figures/confusion_matrix.png")
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, probabilities)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    plt.legend()
    plt.tight_layout()

    plt.savefig("reports/figures/roc_curve.png")
    plt.close()

    print("\nEvaluation figures saved successfully!")


if __name__ == "__main__":
    evaluate_model()
