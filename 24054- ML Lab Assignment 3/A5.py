#LAB 3- A5

import numpy as np
import pandas as pd
from scipy.spatial.distance import minkowski

df=pd.read_csv("DCT_mal (2) 1.csv")
X=df.drop(columns=["LABEL"]).values    #taking only the features (not considering label column, since that is output y)
A=X[4]
B=X[7]    #selecting two feature vectors from the dataset

def minkowski_distance(A,B,p):
    total=0
    for i in range(len(A)):
        total=total+abs(A[i]-B[i])**p   #absolute value of ai and bi raised to the power p
    return total**(1/p)

p=2   #choosing p to be 2 (can be any number)
distance=minkowski_distance(A,B,p)
scipy_distance=minkowski(A,B,p)

print("Function Minkowski Distance:",distance)
print("SciPy Minkowski Distance:",scipy_distance)
