
import joblib
import pandas as pd


def predict_churn(customer_data):

    model = joblib.load("models/best_model.joblib")

    # Convert input into DataFrame
    customer_df = pd.DataFrame([customer_data])

    prediction = model.predict(customer_df)[0]

    probability = model.predict_proba(customer_df)[0][1]

    result = {
        "prediction": "Likely to Churn" if prediction == 1
                      else "Likely to Stay",

        "churn_probability": round(float(probability), 4)
    }

    return result


if __name__ == "__main__":

    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5,
        "TotalCharges": 427.5
    }

    result = predict_churn(sample_customer)

    print(result)
