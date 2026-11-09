
import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

# -----------------------------------
# 1. Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Churn Prediction System")
st.write(
    "Upload customer information to predict which customers "
    "are likely to leave the company."
)

# -----------------------------------
# 2. File Paths
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"

# -----------------------------------
# 3. Check Model Files
# -----------------------------------

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.error(
        "Model files not found. Please train the model first "
        "by running: python src/train.py"
    )
    st.stop()

# -----------------------------------
# 4. Load Model and Metadata
# -----------------------------------

try:
    model = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r") as file:
        metadata = json.load(file)

    required_features = metadata["features"]

except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.stop()

# -----------------------------------
# 5. File Upload
# -----------------------------------

st.subheader("Upload Customer Data")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

# -----------------------------------
# 6. Process Uploaded File
# -----------------------------------

if uploaded_file is not None:

    try:
        input_data = pd.read_csv(uploaded_file)

        st.success("File uploaded successfully!")

        st.subheader("Uploaded Data Preview")
        st.dataframe(input_data.head())

        # Remove target and ID columns if present
        input_data = input_data.drop(
            columns=["Churn", "customerID"],
            errors="ignore"
        )

        # Check for missing required columns
        missing_columns = [
            col for col in required_features
            if col not in input_data.columns
        ]

        if missing_columns:
            st.error(
                "Your CSV is missing these required columns: "
                +