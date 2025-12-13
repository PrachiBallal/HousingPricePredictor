import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from feature_engineering import scale_features
import numpy as np

def evaluate_model():
    y_test = pd.read_csv("data/processed/y_test.csv")

    _, X_test_scaled = scale_features()

    model = joblib.load("models/model.pkl")
    predictions = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"R2 Score: {r2}")

if __name__ == "__main__":
    evaluate_model()
