"""The regression model gives low values of MSE, RMSE and MAPE values with an R2 value of 1.0 in both training and testing data. This means that there is a close relation between the values of payment predicted and those realised.
The model generalizes well for unseen samples/data also.
Therefore, there is no underfitting or overfitting of the model."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

def metrics(actual,pred):
    mse=mean_squared_error(actual,pred)
    rmse=np.sqrt(mse)   #root mean squared
    mape=np.mean(np.abs((actual-pred)/actual))*100   #mean of absolute percentage error
    r2=r2_score(actual,pred)
    return mse,rmse,mape,r2



df=pd.read_excel("Lab Session Data.xlsx", sheet_name="Purchase data")
X=df[['Candies (#)','Mangoes (Kg)','Milk Packets (#)']].values   #separate the features and labels
y=df["Payment (Rs)"].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=7)

m=LinearRegression()      #training the linear regression model
m.fit(X_train,y_train)

pred1=m.predict(X_train)  #predicting the values for training and testing data
pred2=m.predict(X_test)

mse1,rmse1,mape1,r2_1=metrics(y_train,pred1)
mse2,rmse2,mape2,r2_2=metrics(y_test,pred2)

print("Training dataset metrics:")
print("mse:",mse1)
print("rmse:",rmse1)
print("mape:",mape1)
print("r2 score:",r2_1)

print("Testing dataset metrics:")
print("mse:",mse2)
print("rmse:",rmse2)
print("mape:",mape2)
print("r2 score:",r2_2)
