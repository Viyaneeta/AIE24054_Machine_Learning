#LAB 4- A1
"""The model shows regular fit since the performance metrics of the training and testing data are similar/comparable with only a small drop.
The knn classifier achieves high scores on training data and slightly lower for testing data.
Model generalizes well to unseen samples."""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score


def performance(knn,X,y):
    pred=knn.predict(X)
    matrix=confusion_matrix(y,pred)
    precision=precision_score(y,pred,average="weighted") #calculating the weighted average
    recall=recall_score(y,pred,average="weighted")
    f1=f1_score(y,pred,average="weighted")
    return matrix,precision,recall,f1
    

df=pd.read_csv("DCT_mal (2) 1.csv")
X=df.drop(columns=["LABEL"]).values   #separate the features and labels
y=df["LABEL"].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=7)


#training the knn classifier
k=3     #taking 3 neighbors 
knn=KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train,y_train)

#for training data
matrix1,precision1,recall1,f1_1=performance(knn,X_train,y_train)

#for testing data
matrix2,precision2,recall2,f1_2=performance(knn,X_test,y_test)

print("Training data performance metrics:")
print("Confusion matrix:", matrix1)
print("Precision:", precision1)
print("Recall:",recall1)
print("F1 score:", f1_1)

print("Testing data performance metrics:")
print("Confusion matrix:", matrix2)
print("Precision:", precision2)
print("Recall:",recall2)
print("F1 score:", f1_2)
