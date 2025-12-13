from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

# POST route for predictions
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array([list(data.values())])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return jsonify({"predicted_price": float(prediction[0])})

# GET route for testing in browser
@app.route("/", methods=["GET"])
def home():
    return "Flask server is running! Use POST /predict for predictions."

# Start the Flask app
if __name__ == "__main__":
    app.run(debug=True)
