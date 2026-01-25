#Lab 2 - A5

import pandas as pd
import numpy as np

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

def main():

    df = pd.read_excel(r"C:\Users\Viyaneeta Ramesh\OneDrive\Documents\SEM 4\Lab Session Data.xlsx",sheet_name="thyroid0387_UCI")

    vector1=df.iloc[0]        #considering the first 2 observation vectors from the dataset
    vector2=df.iloc[1]

    binary_columns = []
    for col in df.columns:
        values = set(df[col].dropna().unique())    #take columns having t and f 
        if values.issubset({'t', 'f'}):
            binary_columns.append(col)
            
    b1 = vector1[binary_columns].map({'t':1, 'f':0})    #convert t and f into 1 and 0
    b2 = vector2[binary_columns].map({'t':1, 'f':0})


    jc = jaccard_coefficient(b1, b2)
    smc = simple_matching_coefficient(b1, b2)

    print("Jaccard Coefficient:", jc)
    print("Simple Matching Coefficient:", smc)
if __name__ == "__main__":
    main()
