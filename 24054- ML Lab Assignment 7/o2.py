import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer

# load data
df = pd.read_csv("DCT_mal (2) 1.csv")

x = df.drop("LABEL", axis=1)
y = df["LABEL"]

# split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42, stratify=y
)

# scale
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# model
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

# lime explainer
explainer = LimeTabularExplainer(
    x_train,
    mode="classification"
)

# explain one sample
exp = explainer.explain_instance(
    x_test[0],
    model.predict_proba
)

# print explanation
print(exp.as_list())
