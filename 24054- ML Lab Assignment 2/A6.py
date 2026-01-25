#LAB 2 - A6

import pandas as pd
import numpy as np

def cosine_similarity(v1,v2):
    product=np.dot(v1,v2)          #calculate dot product
    magnitude_v1=np.sqrt(np.sum(v1**2))
    magnitude_v2=np.sqrt(np.sum(v2**2))     #calculate magnitude
    return product/(magnitude_v1*magnitude_v2)     #use formula to calculate (a.b/|a|.|b|)


def main():
    df = pd.read_excel(r"C:\Users\Viyaneeta Ramesh\OneDrive\Documents\SEM 4\Lab Session Data.xlsx",sheet_name="thyroid0387_UCI")
    x1=df.iloc[0]
    x2=df.iloc[1]
    
    x1=pd.to_numeric(x1,errors='coerce').fillna(0)  #make he attributes numeric and non numeric becomes 0
    x2=pd.to_numeric(x2,errors='coerce').fillna(0)

    cs=cosine_similarity(x1.values,x2.values)
    print(cs)

if __name__=="__main__":
    main()


