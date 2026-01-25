#LAB 2 - A9

import pandas as pd

def data_scaling(df):
    for i in df.columns:
        if df[i].dtype!="object":  #we consider only numeric datatype and ignore categorical
            Min=df[i].min()
            Max=df[i].max()
            df[i]=(df[i]-Min)/(Max-Min)     #formula for normalization
    return df

def main():
    df = pd.read_excel(r"C:\Users\Viyaneeta Ramesh\OneDrive\Documents\SEM 4\Lab Session Data.xlsx",sheet_name="thyroid0387_UCI")
    print("Before scaling:",df.head())
    x=data_scaling(df)
    print("After scaling:",x.head())

if __name__=="__main__":
    main()
