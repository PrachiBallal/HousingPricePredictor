import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def scale_features():
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, "models/scaler.pkl")

    return X_train_scaled, X_test_scaled

if __name__ == "__main__":
    scale_features()
