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

n=KNeighborsClassifier(n_neighbors=3)
n.fit(X_train,y_train)
predictions=n.predict(X_test)

def confusion_matrix(actual,predicted,pos):   #actual=true labels
    tp=tn=fp=fn=0  #initializing to 0
    for a,p in zip(actual,predicted):
        if a==pos and p==pos:     #model has predicted the actual positive value
            tp+=1
        elif a!=pos and p!=pos:   #model predicted correctly
            tn+=1
        elif a==pos and p!=pos:    #model predicted false when true
            tn+=1
        elif a!=pos and p==pos:    #model predicted true when false
            fn+=1
    return tp,tn,fp,fn

def accuracy(tp,tn,fp,fn):
    return (tp+tn)/(tp+tn+fp+fn)    #according to formula

def precision(tp,fp):
    return tp/(tp+fn)

def recall(tp,fn):
    if tp+fn==0:
        return 0
    return tp/(tp+fn)

def f_score(P,R,B):
    if P+R==0:
        return 0
    return (1+B**2)*P*R/(B**2*P+R)

pos=2
tp,tn,fp,fn=confusion_matrix(y_test,predictions,pos)
print("TP:", tp, "TN:", tn, "FP:", fp, "FN:", fn)
acc=accuracy(tp, tn, fp, fn)
prec=precision(tp,fp)
rec=recall(tp, fn)

B=float(input("Enter beta value for f-beta score"))
fscore=f_score(prec,rec,B)
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F-beta Score:", fscore)
