import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

def load():
    df=pd.read_csv("DCT_mal (2) 1.csv")
    X=df.iloc[:,0:2].values   #separate the features and labels
    y=df["LABEL"].values

    labels=np.unique(y)    #to select any two classes from the dataset
    label1,label2=labels[0],labels[1]

    m=(y==label1)|(y==label2)    #filter data and keep only those two classes
    X=X[m]
    y=y[m]

    y_final=np.where(y==label1,0,1)
    return X,y_final

def hyperparameter():
    X,y=load()
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=54)
    knn=KNeighborsClassifier()   #object for knn classifier
    k_values={"n_neighbors":[1,3,5,7,9,11]}  #range of k values 

    grid=GridSearchCV(knn,k_values,cv=5,scoring="accuracy") #cv=5 for 5 fold cross validation 
    grid.fit(X_train,y_train)

    print("Best k value is:", grid.best_params_["n_neighbors"])
    print("Best k function:", grid.best_score_)

    best=grid.best_estimator_
    accuracy=best.score(X_test,y_test)
    print("Accuracy is:",accuracy)

hyperparameter()
    
