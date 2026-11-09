
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def load_data(path):
    """Load the customer churn dataset."""

    df = pd.read_csv(path)

    # Convert TotalCharges to numeric.
    # Invalid/blank values become NaN.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    return df


def prepare_data(df):
    """Separate features and target."""

    # Remove customer ID because it does not help prediction.
    df = df.drop(columns=["customerID"], errors="ignore")

    # Separate target from input features.
    X = df.drop(columns=["Churn"])
    y = df["Churn"].map({"Yes": 1, "No": 0})

    if y.isna().any():
        raise ValueError("Unexpected or missing target values.")

    return X, y


def build_preprocessor(X):
    """Create numerical and categorical preprocessing pipelines."""

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    return preprocessor
