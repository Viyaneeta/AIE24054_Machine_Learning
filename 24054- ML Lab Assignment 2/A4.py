#LAB 2 - A4

import pandas as pd
import numpy as np
#find data types, numeric range, missing values,mean,variance,attributes

df=pd.read_excel(r"C:\Users\Viyaneeta Ramesh\OneDrive\Documents\SEM 4\Lab Session Data.xlsx",sheet_name="thyroid0387_UCI")
print("Datatypes:", df.dtypes)
print("Numeric Ranges:")
print(df.describe().loc[['min','max']])
print("Missing values:", df.isnull().sum())
print("Mean:",df.mean(numeric_only=True))
print("Variance:",df.var(numeric_only=True))
print("Categorical Attributes:")
print(df.select_dtypes(include='object').columns)
