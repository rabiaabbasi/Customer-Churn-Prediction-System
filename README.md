
# Customer Churn Prediction System

An end-to-end Machine Learning project that predicts
whether a telecom customer is likely to leave a service.

## Project Overview

This project uses historical customer data to train
machine learning models and identify customers at risk
of leaving a telecom company.

## Features

- Data cleaning and preprocessing
- Exploratory Data Analysis
- Feature engineering
- Multiple ML model training
- Model comparison
- Performance evaluation
- Churn probability prediction
- Interactive Streamlit dashboard
- Batch CSV predictions

## Machine Learning Models

- Logistic Regression
- Random Forest
- Gradient Boosting

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## Dataset

IBM Telco Customer Churn Dataset.

Download:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Place the downloaded CSV inside:

data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv

## Installation

Clone the repository:

git clone YOUR_GITHUB_REPOSITORY_URL

Move into the project directory:

cd customer-churn-prediction-system

Create a virtual environment:

python -m venv .venv

Install dependencies:

pip install -r requirements.txt

## Train the Model

python src/train.py

## Evaluate the Model

python src/evaluate.py

## Run the Dashboard

streamlit run app.py

## Project Outputs

- Best trained model
- Model comparison report
- Confusion matrix
- ROC curve
- Customer churn predictions

## Author

Your Name

AI & Machine Learning Developer
