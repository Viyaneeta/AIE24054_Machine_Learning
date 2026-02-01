#LAB 3 - A11


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df=pd.read_csv("DCT_mal (2) 1.csv")
X=df.drop(columns=["LABEL"]).values     #separating features and the labels
y=df["LABEL"].values

x_class=X[(y==3367)|(y==2)] #selecting only two classes for binary classification
y_class=y[(y==3367)|(y==2)]

X_train,X_test,y_train,y_test=train_test_split(x_class,y_class,test_size=0.3)   #30% for testing purpose

knn1=KNeighborsClassifier(n_neighbors=1)
knn1.fit(X_train,y_train)        #for k=1 nearest neighbor
accuracy1=knn1.score(X_test,y_test)

knn3=KNeighborsClassifier(n_neighbors=3)
knn3.fit(X_train,y_train)        #for k=3 nearest neighbor
accuracy3=knn3.score(X_test,y_test)

print("Accuracy for k=1:",accuracy1,"   Accuracy for k=3 is:",accuracy3)

krange=range(1,12)   #varying k from 1 to 11
accuracies=[]   #empty list to append and for plotting

for k in krange:
        model=KNeighborsClassifier(n_neighbors=k) #for each value of k, fit in model
        model.fit(X_train,y_train)
        model_accuracy=model.score(X_test,y_test)
        accuracies.append(model_accuracy)
plt.plot(krange,accuracies,marker='o')
plt.title("k vs Accuracy")
plt.show()
                          
