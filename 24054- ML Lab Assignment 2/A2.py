#LAB 2 - Q2

import pandas as pd
from sklearn.model_selection import train_test_split    
from sklearn.linear_model import LogisticRegression

df=pd.read_excel(r"C:\Users\Viyaneeta Ramesh\OneDrive\Documents\SEM 4\Lab Session Data.xlsx",sheet_name="Purchase data")

df['Label']=df['Payment (Rs)'].apply(lambda x:'rich' if x>200 else 'poor')   #apply condition and label
X=df[['Candies (#)','Mangoes (Kg)','Milk Packets (#)']]
y=df['Label']


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)  #30% for testing with any random state

model=LogisticRegression()
model.fit(X_train,y_train)      #train the model
prediction=model.predict(X_test)    


df['Class']=model.predict(X)
print(df[['Candies (#)','Mangoes (Kg)','Milk Packets (#)','Class']])
