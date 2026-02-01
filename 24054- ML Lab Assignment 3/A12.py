#LAB 3 - A12

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df=pd.read_csv("DCT_mal (2) 1.csv")
X=df.drop(columns=["LABEL"]).values     #separating features and the labels
y=df["LABEL"].values

x_class=X[(y==3367)|(y==2)] #selecting only two classes for binary classification
y_class=y[(y==3367)|(y==2)]

X_train,X_test,y_train,y_test=train_test_split(x_class,y_class,test_size=0.3)   #30% for testing purpose

n=KNeighborsClassifier(n_neighbors=3)
n.fit(X_train,y_train)

training_prediction=n.predict(X_train)
training_matrix=confusion_matrix(y_train,training_prediction)
print("Confusion matrix for training data:",training_matrix)
training_precision=precision_score(y_train,training_prediction,average="binary",pos_label=2) #taking the positive label to be class 2, with binary classification
training_recall=recall_score(y_train,training_prediction,average="binary",pos_label=2)
training_f1=f1_score(y_train,training_prediction,average="binary",pos_label=2)
print("Training Metrics:")
print("Precision:",training_precision)
print("Recall:",training_recall)
print("F1 Score:",training_f1)



testing_prediction=n.predict(X_test)
testing_matrix=confusion_matrix(y_test,testing_prediction)
print("Confusion matrix for testing data:",testing_matrix)
testing_precision = precision_score(y_test, testing_prediction, average="binary", pos_label=2)   
testing_recall = recall_score(y_test, testing_prediction, average="binary", pos_label=2)
testing_f1 = f1_score(y_test, testing_prediction, average="binary", pos_label=2)
print("Testing Metrics:")
print("Precision:", testing_precision)
print("Recall:", testing_recall)
print("F1 Score:", testing_f1)

