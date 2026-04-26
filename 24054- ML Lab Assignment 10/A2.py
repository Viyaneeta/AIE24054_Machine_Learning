import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_data(path):
    data = pd.read_csv(path)
    x = data.drop(columns=["LABEL"])
    y = data["LABEL"]
    return x, y

def preprocess(x):
    sc = StandardScaler()
    return sc.fit_transform(x)

def apply_pca(x, var):
    pca = PCA(n_components=var)
    return pca.fit_transform(x)

def split_data(x, y):
    return train_test_split(x, y, test_size=0.2, random_state=42)

def tune_svm(x_train, y_train):
    params = {'C':[0.1,1,10], 'kernel':['rbf','linear']}
    grid = GridSearchCV(SVC(probability=True), params, cv=3, n_jobs=-1)
    grid.fit(x_train, y_train)
    return grid.best_estimator_

def tune_rf(x_train, y_train):
    params = {'n_estimators':[100,200], 'max_depth':[10,None]}
    grid = GridSearchCV(RandomForestClassifier(), params, cv=3, n_jobs=-1)
    grid.fit(x_train, y_train)
    return grid.best_estimator_

def evaluate(model, x_test, y_test):
    y_pred = model.predict(x_test)
    return accuracy_score(y_test, y_pred)

# main
path = 'DCT_mal (2) 1.csv'

x, y = load_data(path)
x_scaled = preprocess(x)

x_pca = apply_pca(x_scaled, 0.99)

x_train, x_test, y_train, y_test = split_data(x_pca, y)

svm_model = tune_svm(x_train, y_train)
rf_model = tune_rf(x_train, y_train)

print('pca 99 svm:', evaluate(svm_model, x_test, y_test))
print('pca 99 rf:', evaluate(rf_model, x_test, y_test))
print('reduced features:', x_pca.shape[1])
