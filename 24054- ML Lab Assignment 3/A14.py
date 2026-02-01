import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df=pd.read_csv("DCT_mal (2) 1.csv")
X=df.drop(columns=["LABEL"]).values     #separating features and the labels
y=df["LABEL"].values

x_class=X[(y==3367)|(y==2)] #selecting only two classes for binary classification
y_class=y[(y==3367)|(y==2)]
y_binary=np.where(y_class==2,1,0)

X_train,X_test,y_train,y_test=train_test_split(x_class,y_binary,test_size=0.3)   #30% for testing purpose

n=KNeighborsClassifier(n_neighbors=3)
n.fit(X_train,y_train)
predictions=n.predict(X_test)
accuracy=np.mean(predictions==y_test)
print("KNN ACCURACY:", accuracy)

#W=inv(XTX) XTy
X_train_bias = np.c_[np.ones(len(X_train)), X_train]
X_test_bias = np.c_[np.ones(len(X_test)), X_test]

XT = X_train_bias.T
XTX = np.dot(XT, X_train_bias)    #forming the equation using numpy functions
XTX_inv = np.linalg.inv(XTX)
XTy = np.dot(XT, y_train)

W = np.dot(XTX_inv, XTy)


y_pred= np.dot(X_test_bias, W)


y_pred_matrix = []

for val in y_pred:
    if val>=0.5:
        y_pred_matrix.append(1)
    else:
        y_pred_matrix.append(0)

y_pred_matrix = np.array(y_pred_matrix)
matrix_accuracy = np.mean(y_pred_matrix==y_test)
print("Matric inversion accuracy:",matrix_accuracy)
