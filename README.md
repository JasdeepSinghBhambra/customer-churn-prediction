# Customer Churn Prediction

A machine learning web application that predicts whether a customer is likely to churn based on customer account information, service usage, and calling behavior.

## Live Demo

https://customer-churn-prediction-aedr.onrender.com

## Project Overview

Customer churn refers to customers leaving or discontinuing a service.

This project uses machine learning to predict customer churn from historical customer data.

The application provides a simple web interface where users can enter customer information and receive a churn prediction.

## Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Logistic Regression
- HTML
- CSS
- JavaScript
- Git
- GitHub
- Render
- Gunicorn

## Machine Learning Workflow

1. Load customer churn dataset
2. Handle missing values
3. Encode categorical variables
4. Scale numerical features
5. Train a Logistic Regression model
6. Receive customer information from the web form
7. Preprocess the input
8. Generate churn prediction
9. Display the prediction to the user

## Project Structure

```text
customer-churn-prediction/
│
├── app.py
├── churn.csv
├── churn_model.pkl
├── requirements.txt
├── 1.ipynb
├── README.md
│
└── templates/
    └── index.html
