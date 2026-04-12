import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
#load data
def ld(fp):
    d = pd.read_csv(fp)
    x = d.iloc[:, :-1].values
    y = d.iloc[:, -1].values
    return x, y
#split data
def sp(x, y):
    return train_test_split(x, y, test_size=0.2)

#scale data
def scd(xtr, xts):
    sc = StandardScaler()
    xtr = sc.fit_transform(xtr)
    xts = sc.transform(xts)
    return xtr, xts

#hyperparameter tuning - tune model
def tune(xtr, ytr):
    m = MLPClassifier(max_iter=1000)
    p = {
        'hidden_layer_sizes': [(50,), (100,), (50,50)],
        'learning_rate_init': [0.001, 0.01],
        'activation': ['relu', 'tanh']
    }
    g = GridSearchCV(m, p, cv=3)
    g.fit(xtr, ytr)
    return g

#evaluate
def ev(m, xts, yts):
    yp = m.predict(xts)
    return accuracy_score(yts, yp)

# main
x, y = ld("DCT_mal (2) 1.csv")
xtr, xts, ytr, yts = sp(x, y)
xtr, xts = scd(xtr, xts)
g = tune(xtr, ytr)
bm = g.best_estimator_
acc = ev(bm, xts, yts)

print("best params:", g.best_params_)
print("accuracy:", acc)
