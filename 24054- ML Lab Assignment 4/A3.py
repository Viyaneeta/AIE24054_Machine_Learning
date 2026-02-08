import numpy as np
import matplotlib.pyplot as plt

np.random.seed(54)

X=np.random.randint(1,11,size=(20,2))   #to get 20 data spoint with 2 features
y=np.random.randint(0,2,size=20)  #it randomly assigns a class label fot the 20 points, either 0 or 1

c0=X[y==0]
c1=X[y==1]      #separating them based on their class

plt.scatter(c0[:,0],c0[:,1],color="blue",label="class 0")   #class zero is blue
plt.scatter(c1[:,0],c1[:,1],color="red",label="class 1")   #class one is red

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Scatter plot of Training data")
plt.legend()
plt.show()
