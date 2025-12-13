from sklearn.datasets import fetch_california_housing
import pandas as pd
import os

def ingest_data():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame

    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/housing.csv", index=False)

    print("Data ingestion completed")

if __name__ == "__main__":
    ingest_data()
