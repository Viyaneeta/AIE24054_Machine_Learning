import numpy as np
import pandas as pd

df=pd.read_csv("DCT_mal (2) 1.csv")
X=df.drop(columns=["LABEL"]).values #feature/observational vector
y=df["LABEL"].values     #output values

c1=X[y==1]
c2=X[y==2]    #choosing two classes 1 and 2

def mean(data):
    return np.sum(data,axis=0)/data.shape[0]   #sum column wise and divided by total number of samples

def variance(data):
    mean1=mean(data)
    return np.sum((data-mean1)**2, axis=0)/data.shape[0]  #taking average of square of difference

def standard_deviation(data):
    return np.sqrt(variance(data))   #std is the sqrt of variance (sigma)

def euclidean_norm(x):
    sum=0
    for i in range(len(x)):
        sum=sum+x[i]**2
    return np.sqrt(sum)    #square root of sum of square of each element

mean_1=mean(c1)
mean_2=mean(c2)
std1= standard_deviation(c1)
std2= standard_deviation(c2)
interclass_distance=euclidean_norm(mean_1-mean_2)

print("Centroid of class 1:",mean_1)
print("Centroid of class 2:",mean_2)
print("Standard deviation of class 1:",std1)
print("Standard deviation of class 1:",std2)
print("Interclass distance between class 1 and class2:", interclass_distance)


