#LAB 2 - A8

import pandas as pd

def impute(df):
    for i in df.columns:
        if df[i].dtype=="object":       #if it is categorical data then we
            df[i]=df[i].fillna(df[i].mode()[0]) #fill the missing values using the mode function
        else:
            if df[i].skew()<1:         #no outliers=lower skew value
                df[i]=df[i].fillna(df[i].mean()) #filled the missing values with mean according to q
            else:
                df[i]=df[i].fillna(df[i].median()) #if many outliers are present fill missing values with median
    return df

def main():
    df = pd.read_excel(r"C:\Users\Viyaneeta Ramesh\OneDrive\Documents\SEM 4\Lab Session Data.xlsx",sheet_name="thyroid0387_UCI")
    print("No imputation:",df.isnull().sum())

    x=impute(df)
    print("After Data imputation:",x.isnull().sum())

if __name__=="__main__":
    main()
