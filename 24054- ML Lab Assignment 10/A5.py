import pandas as pd
import lime.lime_tabular
import shap
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def load_data(path):
    data = pd.read_csv(path)
    x = data.drop(columns=["LABEL"])
    y = data["LABEL"]
    return x, y

def preprocess(x):
    sc = StandardScaler()
    return sc.fit_transform(x)

def split_data(x, y):
    return train_test_split(x, y, test_size=0.2, random_state=42)

def train_model(x_train, y_train):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(x_train, y_train)
    return model

#lime
def lime_explain(model, x_train, x_test):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=x_train,
        mode='classification'
    )
    
    exp = explainer.explain_instance(
        x_test[0],
        model.predict_proba
    )
    
    print('\nlime explanation:')
    print(exp.as_list())

#shap
def shap_explain(model, x_sample):
    explainer = shap.Explainer(model, x_sample)
    shap_values = explainer(x_sample)

    print('\nshap summary plot:')
    shap.plots.bar(shap_values)

    plt.show()

#main
path = 'DCT_mal (2) 1.csv'

x, y = load_data(path)
x_scaled = preprocess(x)

x_train, x_test, y_train, y_test = split_data(x_scaled, y)

model = train_model(x_train, y_train)

lime_explain(model, x_train, x_test)
shap_explain(model, x_train[:100])
