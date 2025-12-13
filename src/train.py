import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from feature_engineering import scale_features

def train_model():
    y_train = pd.read_csv("data/processed/y_train.csv")

    X_train_scaled, _ = scale_features()

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train_scaled, y_train.values.ravel())

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    print("Model training completed")

if __name__ == "__main__":
    train_model()
