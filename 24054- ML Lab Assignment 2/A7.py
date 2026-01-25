#Lab 2 - A7

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def jaccard_coefficient(v1, v2): #JC = (f11/f01+f10+f11)
    
    f01=np.sum((v1==0)&(v2==1))
    f10=np.sum((v1==1)&(v2==0))
    f11=np.sum((v1==1)&(v2==1))
    return f11/(f01+f10+f11) if (f01+f10+f11) !=0 else 0

def simple_matching_coefficient(v1, v2): #SMC = (f11 + f00) / (f00 +f01 + f10 + f11)
    f01=np.sum((v1==0)&(v2==1))
    f10=np.sum((v1==1)&(v2==0))
    f11=np.sum((v1==1)&(v2==1))
    f00=np.sum((v1==0)&(v2==0))
    denom=f00+f01+f10+f11
    return (f11+f00)/denom if denom!=0 else 0

def cosine_similarity(v1,v2):
    product=np.dot(v1,v2)          #calculate dot product
    magnitude_v1=np.sqrt(np.sum(v1**2))
    magnitude_v2=np.sqrt(np.sum(v2**2))     #calculate magnitude
    return product/(magnitude_v1*magnitude_v2)     #use formula to calculate (a.b/|a|.|b|)

def main():

    df = pd.read_excel(r"C:\Users\Viyaneeta Ramesh\OneDrive\Documents\SEM 4\Lab Session Data.xlsx",sheet_name="thyroid0387_UCI")

    binary_columns = []
    for col in df.columns:
        values = set(df[col].dropna().unique())    #take columns having t and f 
        if values.issubset({'t', 'f'}):
            binary_columns.append(col)
            
    b1=df[binary_columns].replace({'t':1, 'f':0})    #convert t and f into 1 and 0
    binary=b1.head(20).values
    numeric=df.head(20).apply(pd.to_numeric, errors='coerce').fillna(0).values

    n=20
    jc=np.zeros((n,n))
    smc=np.zeros((n,n))
    cos=np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            jc[i,j]=jaccard_coefficient(binary[i],binary[j])
            smc[i,j]=simple_matching_coefficient(binary[i],binary[j])
            cos[i,j]=cosine_similarity(numeric[i],numeric[j])
    sns.heatmap(jc)
    plt.title("Jaccard Coefficient")
    plt.show()
 
    sns.heatmap(smc)
    plt.title("Simple Matching Coefficient")
    plt.show()

    sns.heatmap(cos)
    plt.title("Cosine Similarity")
    plt.show()




if __name__ == "__main__":
    main()
