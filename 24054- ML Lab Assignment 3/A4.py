#LAB 3 - A4

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("DCT_mal (2) 1.csv")
X=df.drop(columns=["LABEL"]).values    #taking only the features (not considering label column, since that is output y)
A=X[4]
B=X[7]    #selecting two feature vectors from the dataset

def minkowski_distance(A,B,p):
    total=0
    for i in range(len(A)):
        total=total+abs(A[i]-B[i])**p   #absolute value of ai and bi raised to the power p
    return total**(1/p)   #pth root of the sum found (total)

prange=range(1,11)
dist1=[]    #taking empty list to append all distances

for p in prange:
    dist=minkowski_distance(A,B,p)
    dist1.append(dist)

plt.plot(prange,dist1,marker='o')    #dist decreases as the value of p increases
plt.title("Miskowski distance")
plt.show()

for p,d in zip(prange,dist1):
    print(f"Minkowski distance for p={p}: {d}")
