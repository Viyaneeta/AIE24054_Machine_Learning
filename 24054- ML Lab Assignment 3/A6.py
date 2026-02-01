#LAB 3 - A6

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df=pd.read_csv("DCT_mal (2) 1.csv")
X=df.drop(columns=["LABEL"]).values
y=df["LABEL"].values

x_class=X[(y==1)|(y==2)] #selecting only two classes for binary classification
y_class=y[(y==1)|(y==2)]

X_train,X_test,y_train,y_test=train_test_split(x_class,y_class,test_size=0.3)   #30% for testing purpose

print("Total:",len(x_class))
print("Training samples:",len(X_train))
print("Testing samples:",len(X_test))
