from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings
import os

warnings.filterwarnings("ignore")

app = Flask(__name__)

# =========================
# LOAD DATA
# =========================

data = pd.read_csv("churn.csv")
raw = pd.read_csv("churn.csv")

# Handle missing numeric values
data.fillna(data.mean(numeric_only=True), inplace=True)
raw.fillna(raw.mean(numeric_only=True), inplace=True)

# =========================
# STATE OPTIONS
# =========================

try:
    state_options = sorted(
        raw["State"].dropna().unique().tolist()
    )
except Exception:
    state_options = []

# =========================
# NUMERIC LIMITS
# =========================

from pandas.api import types as pdtypes

limits = {}

for col in raw.columns:

    if col == "Churn?":
        continue

    try:
        if pdtypes.is_numeric_dtype(raw[col]):

            col_min = raw[col].min()
            col_max = raw[col].max()

            if float(col_min).is_integer() and float(col_max).is_integer():
                col_min = int(col_min)
                col_max = int(col_max)

            limits[col] = {
                "min": col_min,
                "max": col_max
            }

    except Exception:
        continue

# =========================
# ENCODE CATEGORICAL DATA
# =========================

import pandas.api.types as ptypes

encoders = {}

for col in data.columns:

    if (
        (data[col].dtype == "object"
         or ptypes.is_string_dtype(data[col]))
        and col != "Churn?"
    ):

        le = LabelEncoder()

        data[col] = le.fit_transform(
            data[col].astype(str)
        )

        encoders[col] = le

# =========================
# ENCODE TARGET
# =========================

target_le = LabelEncoder()

data["Churn?"] = target_le.fit_transform(
    data["Churn?"].astype(str)
)

# =========================
# FEATURES AND TARGET
# =========================

X = data.drop("Churn?", axis=1)
y = data["Churn?"]

# =========================
# SCALE FEATURES
# =========================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# =========================
# TRAIN MODEL
# =========================

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_scaled, y)

# =========================
# HOME PAGE
# =========================

@app.route("/")
def index():

    return render_template(
        "index.html",
        state_options=state_options,
        limits=limits
    )

# =========================
# PREDICTION
# =========================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get data from form or JSON
        req_data = request.get_json(silent=True)

        if not req_data:
            req_data = request.form.to_dict()

        # Create dataframe
        input_df = pd.DataFrame([req_data])

        # Keep same column order as training data
        input_df = input_df.reindex(
            columns=X.columns
        )

        # =========================
        # PROCESS INPUTS
        # =========================

        for col in input_df.columns:

            # Categorical columns
            if col in encoders:

                le = encoders[col]

                known_classes = set(
                    str(c) for c in le.classes_
                )

                value = input_df[col].iloc[0]

                if pd.isna(value):
                    value = str(le.classes_[0])

                if str(value) not in known_classes:
                    value = str(le.classes_[0])

                input_df[col] = [
                    le.transform([str(value)])[0]
                ]

            # Numeric columns
            else:

                input_df[col] = pd.to_numeric(
                    input_df[col],
                    errors="coerce"
                )

                input_df[col] = input_df[col].fillna(
                    X[col].median()
                )

        # =========================
        # SCALE INPUT
        # =========================

        input_scaled = scaler.transform(input_df)

        # =========================
        # PREDICT
        # =========================

        prediction = model.predict(input_scaled)

        result_class = target_le.inverse_transform(
            prediction
        )[0]

        # Determine churn
        is_churn = (
            1 if "True" in str(result_class) else 0
        )

        # Probability
        probabilities = model.predict_proba(
            input_scaled
        )[0]

        confidence = float(
            max(probabilities) * 100
        )

        return jsonify({
            "prediction": is_churn,
            "confidence": round(confidence, 2)
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 400


# =========================
# RUN APPLICATION
# =========================

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)