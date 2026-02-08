import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


def a4knn():
    np.random.seed(54)
    X=np.random.randint(1,11,size=(20,2))   #to get 20 data spoint with 2 features
    y_train=np.random.randint(0,2,size=20)  #it randomly assigns a class label fot the 20 points, either 0 or 1

    k=3
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X,y_train)

    x=np.arange(0,10,0.1)      #as per question, varying the value of X and y
    y=np.arange(0,10,0.1)      #100x100 grid
    x1,y1=np.meshgrid(x,y)

    x_s=x1.ravel()  #x values into a single column/ flattening
    y_s=y1.ravel()  #y values into a single column
    X_test=np.column_stack((x_s,y_s))   #joining them to form test data

    prediction=knn.predict(X_test)
        
    plt.scatter(X_test[prediction==0,0],X_test[prediction==0,1],color="blue",s=4,label="Class 0")
    plt.scatter(X_test[prediction==1,0],X_test[prediction==1,1],color="red",s=4,label="Class 1")
    plt.scatter(X[:,0],X[:,1],color="black",marker="x",label="Training data")

    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("knn boundary decision (k=3)")
    plt.legend()
    plt.show()

a4knn()
