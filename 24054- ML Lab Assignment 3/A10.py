#LAB 3 - A10

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

def euclidean_dist(A,B):
    sum=0
    for i in range(len(A)):
        sum=sum+(A[i]-B[i])**2
    return np.sqrt(sum)    #square root of sum of square of each element

def knn(X_t,y_t,point,k):
    distances=[]
    for i in range (len(X_t)):
        d=euclidean_dist(X_t[i],point)  #distance of point from all samples
        distances.append((d,y_t[i]))
    distances.sort(key=lambda x:x[0])    #sorting all the distance is ascending order

    neighbors=distances[:k]    #taking the k nearest neighbors from the list
    votes={}

    for _,label in neighbors:
        votes[label]=votes.get(label,0)+1    #checking for majority vote
    return int(max(votes,key=votes.get)) 
        
k=3
predict=[]
for point in X_test:
    pred=knn(X_train,y_train,point,k)
    predict.append(pred)

neigh=KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train,y_train)
predictions=neigh.predict(X_test)    #taking command from lab3 sheet


print("knn function prediction:",predict[:20])   #printing only first 20 in my case
print("sklearn prediction:",predictions[:20])
