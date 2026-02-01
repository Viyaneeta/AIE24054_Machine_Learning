#LAB 3 - A9

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df=pd.read_csv("DCT_mal (2) 1.csv")
X=df.drop(columns=["LABEL"]).values     #separating features and the labels
y=df["LABEL"].values

x_class=X[(y==3367)|(y==2)] #selecting only two classes for binary classification
y_class=y[(y==3367)|(y==2)]

X_train,X_test,y_train,y_test=train_test_split(x_class,y_class,test_size=0.3)   #30% for testing purpose

n=KNeighborsClassifier(n_neighbors=3)   #taking number of neighbours as 3 k=3
n.fit(X_train,y_train)

predicted_labels=n.predict(X_test)    #taking command from lab3 sheet
print("Predicted labels for the test dataset is:", predicted_labels)

test=X_test[4]   #taking any vector from the testing set
single=n.predict([test])
print("Prediction for one vector from test set:", single)
