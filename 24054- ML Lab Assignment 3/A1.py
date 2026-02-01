#LAB 3 - A1

import numpy as np
import pandas as pd

df=pd.read_csv("DCT_mal (2) 1.csv")
X=df.drop(columns=["LABEL"]).values    #taking only the features (not considering label column, since that is output y)
A=X[4]
B=X[10]     #choosing any two vectors 

def dot_product(A,B):
    result=0
    for i in range(len(A)):  #for any dimension n
        result=result+A[i]*B[i]
    return result

def euclidean_norm(A):
    sum=0
    for i in range(len(A)):
        sum=sum+A[i]**2   #sum of the squares
    return np.sqrt(sum)   #square root according to formula

print("Dot product:", dot_product(A,B))
print("Dot product (numpy):", np.dot(A,B))

print("Euclidean norm(A):",euclidean_norm(A))
print("Euclidean norm numpy :",np.linalg.norm(A))
print("Euclidean norm(B):",euclidean_norm(B))
print("Euclidean norm numpy:",np.linalg.norm(B))
