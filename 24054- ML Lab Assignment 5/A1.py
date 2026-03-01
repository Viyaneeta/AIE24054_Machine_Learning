# project is based on classification
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("DCT_mal (2) 1.csv")
X = df.drop(columns=["LABEL"])   # indexing the data and labels 
y = df["LABEL"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#using one feature
X_train_single = X_train[["0"]]
X_test_single  = X_test[["0"]]

def train_linear_regression(X_train, y_train):
    model = LinearRegression()    # using linear regression model
    model.fit(X_train, y_train)
    return model

model = train_linear_regression(X_train_single, y_train)
y_train_pred = model.predict(X_train_single)
y_test_pred  = model.predict(X_test_single)

print("Train predictions (first 5):", y_train_pred[:5])
print("Test predictions  (first 5):", y_test_pred[:5])
#using a single DCT feature for linear regression gave poor predictions
