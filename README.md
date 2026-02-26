# 🏠 House Price Predictor

An end-to-end Machine Learning project that predicts house prices using a **Random Forest Regressor**.  
This repository follows a modular, production-style ML pipeline including data ingestion, preprocessing, feature engineering, model training, evaluation, and prediction serving.

---

## 📌 Project Overview

This project demonstrates a complete supervised machine learning workflow:

- Data ingestion
- Data preprocessing
- Feature engineering
- Model training
- Model evaluation
- Model serialization
- Prediction interface

The focus is on clean architecture and maintainable ML code structure rather than notebook-only experimentation.

---

## 🗂 Repository Structure

```
house-price-predictor/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── model.pkl
│
├── notebooks/
│
├── src/
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Tech Stack

- Python 3.x
- pandas
- numpy
- scikit-learn
- joblib

---

## 🔄 Machine Learning Pipeline

### 1️⃣ Data Ingestion (`src/data_ingestion.py`)
- Loads raw dataset
- Splits into training and testing sets
- Saves processed datasets

### 2️⃣ Data Preprocessing (`src/data_preprocessing.py`)
- Handles missing values
- Encodes categorical variables
- Cleans and structures dataset

### 3️⃣ Feature Engineering (`src/feature_engineering.py`)
- Applies feature scaling
- Transforms input variables
- Prepares feature matrix

### 4️⃣ Model Training (`src/train.py`)
- Loads processed training data
- Trains a `RandomForestRegressor`
- Saves trained model to:

```
models/model.pkl
```

Run training:
```
python src/train.py
```

### 5️⃣ Model Evaluation (`src/evaluate.py`)
- Loads trained model
- Evaluates on test data
- Calculates:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R² Score

Run evaluation:
```
python src/evaluate.py
```

### 6️⃣ Prediction Interface (`app.py`)
- Loads trained model
- Accepts input features
- Returns predicted house price

Run application:
```
python app.py
```

---

## 📊 Model Used

### Random Forest Regressor

- Ensemble learning method
- Handles non-linear relationships
- Reduces overfitting compared to single decision trees
- Well-suited for tabular datasets

---

## 🚀 Installation

Clone the repository:

```
git clone https://github.com/your-username/house-price-predictor.git
cd house-price-predictor
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## 📈 Future Improvements

- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- Cross-validation pipeline
- Model comparison (XGBoost, LightGBM)
- REST API deployment using FastAPI
- Docker containerization
- CI/CD integration

---

## 🎯 Learning Objectives

This project demonstrates:

- Modular ML architecture
- Clean code organization
- Model persistence
- Reproducible workflows
- Scalable ML pipeline design

---

## 📜 License

This project is intended for educational and portfolio purposes.