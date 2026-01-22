from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("model/breast_cancer_model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form values (5 features)
        radius_mean = float(request.form["radius_mean"])
        texture_mean = float(request.form["texture_mean"])
        perimeter_mean = float(request.form["perimeter_mean"])
        area_mean = float(request.form["area_mean"])
        concavity_mean = float(request.form["concavity_mean"])

        # Arrange features in the SAME order used in training
        feature_arr = np.array([[
            radius_mean,
            texture_mean,
            perimeter_mean,
            area_mean,
            concavity_mean
        ]])

        # Scale features
        features_scaled = scaler.transform(feature_arr)

        # Predict
        prediction = model.predict(features_scaled)[0]

        result = "Malignant" if prediction == 1 else "Benign"

        return render_template(
            "index.html",
            prediction_text=f"Tumor Prediction: {result}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )

if __name__ == "__main__":
    app.run()

