import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

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
    return X,y_final,label1,label2

def scatter_plot():
    X,y,label1,label2=load()
    
    plt.scatter(X[y==0,0],X[y==0,1],color="blue",s=4,label=str(label1))
    plt.scatter(X[y==1,0],X[y==1,1],color="red",s=4,label=str(label2))
    
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Scatter Plot ")
    plt.legend()
    plt.show()

def decision_boundary():
    X,y,label1,label2=load()
    knn=KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)

    x=np.linspace(X[:, 0].min(),X[:, 0].max(),150)
    y_vals=np.linspace(X[:, 1].min(),X[:, 1].max(),150)
    x1, y1=np.meshgrid(x, y_vals)
    X_test=np.column_stack((x1.ravel(), y1.ravel()))
    prediction=knn.predict(X_test)

    
    plt.scatter(X_test[prediction==0,0],X_test[prediction==0,1],color="blue",s=5,label=str(label1))
    plt.scatter(X_test[prediction==1,0],X_test[prediction==1,1],color="red",s=5,label=str(label2))
    plt.scatter(X[:,0],X[:,1],color="black",marker="x",label="Training data")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("KNN Boundary Decision(k=3) ")
    plt.legend()
    plt.show()



def k_values():
    X,y,label1,label2=load()
    krange=[1,3,7]
    
    x=np.linspace(X[:, 0].min(),X[:, 0].max(),150)
    y_vals=np.linspace(X[:, 1].min(),X[:, 1].max(),150)
    x1, y1=np.meshgrid(x, y_vals)
    X_test=np.column_stack((x1.ravel(), y1.ravel()))
    
    for k in krange:
        plt.figure()

        knn=KNeighborsClassifier(n_neighbors=k)
        knn.fit(X,y)

        prediction=knn.predict(X_test)
        
        plt.scatter(X_test[prediction==0,0],X_test[prediction==0,1],color="blue",s=5,label=str(label1))
        plt.scatter(X_test[prediction==1,0],X_test[prediction==1,1],color="red",s=5,label=str(label2))
        plt.scatter(X[:,0],X[:,1],color="black",marker="x",label="Training data")

        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title(f"knn boundary decision (k={k})")
        plt.legend()
        plt.show(block=False)   #added this to show all three plots for values k=1,3,7 one after the other
        plt.pause(2)
        plt.close()

scatter_plot()
decision_boundary()
k_values()

