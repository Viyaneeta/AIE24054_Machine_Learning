#performance metrics
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("DCT_mal (2) 1.csv")
X = df.drop(columns=["LABEL"])
y = df["LABEL"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train_single = X_train[["0"]]
X_test_single  = X_test[["0"]]

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def regression_metrics(y_true, y_pred):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    #avoiding divide by zero since some labels are 0
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-9, y_true))) * 100
    r2   = r2_score(y_true, y_pred)
    return mse, rmse, mape, r2

model = train_linear_regression(X_train_single, y_train)
y_train_pred = model.predict(X_train_single)
y_test_pred  = model.predict(X_test_single)

mse_tr, rmse_tr, mape_tr, r2_tr = regression_metrics(y_train, y_train_pred)
mse_te, rmse_te, mape_te, r2_te = regression_metrics(y_test,  y_test_pred)

print("Train Metrics:")
print(f"MSE:{mse_tr:.4f}")
print(f"RMSE:{rmse_tr:.4f}")
print(f"MAPE:{mape_tr:.4f}%")
print(f"R2:{r2_tr:.4f}")

print("Test Metrics:")
print(f"MSE:{mse_te:.4f}")
print(f"RMSE:{rmse_te:.4f}")
print(f"MAPE:{mape_te:.4f}%")
print(f"R2:{r2_te:.4f}")
