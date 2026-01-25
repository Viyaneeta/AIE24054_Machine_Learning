#LAB 2- Q1

import pandas as pd
import numpy as np

def find_rank(feature_mat):
    return np.linalg.matrix_rank(feature_mat)    #3 linearly independent rows/columns

def calculate_cost(feature_mat,output):
    p_inv=np.linalg.pinv(feature_mat)    #y=aX
    return p_inv.dot(output)             # a=Xinv*y

def main():
    df=pd.read_excel(r"C:\Users\Viyaneeta Ramesh\OneDrive\Documents\SEM 4\Lab Session Data.xlsx",sheet_name="Purchase data")
    X=df[['Candies (#)','Mangoes (Kg)','Milk Packets (#)']].values
    y=df['Payment (Rs)'].values

    rank1=find_rank(X)
    cost=calculate_cost(X,y)     #function calls
    print("Rank:",rank1)
    print("Candy cost:",cost[0])
    print("Mango cost:",cost[1])
    print("Milk Packet cost:",cost[2])

if __name__=="__main__":
    main()
    
"""
Dimensions
All the observations have three characteristics namely candies, mangoes, and milk packets.
The vectors therefore exist in a three dimensional space (R3).

Count of Vectors 
The data is purchase records of 10 customers.
Consequently, the vectors in the vector space are 10.

Rank of Matrix 
The rank of a matrix is the amount of columns or rows that are linearly independent.
The feature matrix has 3 independent columns. """
