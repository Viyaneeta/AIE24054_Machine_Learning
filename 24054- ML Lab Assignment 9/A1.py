#a1
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def load(path):
    df = pd.read_csv("DCT_mal (2) 1")
    return df.drop('LABEL', axis=1).values, df['LABEL'].values

def tune(clf, params, x, y):
    search = RandomizedSearchCV(clf, params, n_iter=10, cv=3, n_jobs=-1, random_state=42)
    search.fit(x, y)
    return search.best_estimator_

def build_stacking(meta, x_tr, y_tr):
    rf_params  = {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]}
    knn_params = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
    svc_params = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}

    base = [('rf',  tune(RandomForestClassifier(random_state=42), rf_params, x_tr, y_tr)),('knn', tune(KNeighborsClassifier(), knn_params, x_tr, y_tr)),('svc', tune(SVC(kernel='rbf', probability=True, random_state=42), svc_params, x_tr, y_tr)),]
    return StackingClassifier(estimators=base, final_estimator=meta, cv=5, n_jobs=-1)

def evaluate(model, x_tr, y_tr, x_te, y_te):
    model.fit(x_tr, y_tr)
    return accuracy_score(y_tr, model.predict(x_tr)), accuracy_score(y_te, model.predict(x_te))


if __name__ == '__main__':
    x, y = load("DCT_mal (2) 1")
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler()
    x_tr, x_te = sc.fit_transform(x_tr), sc.transform(x_te)

    meta_params = {'C': [0.1, 1, 10], 'max_iter': [1000]}
    metas = {'logistic':      tune(LogisticRegression(), meta_params, x_tr, y_tr),'random_forest': tune(RandomForestClassifier(random_state=42), {'n_estimators': [50, 100], 'max_depth': [None, 10]}, x_tr, y_tr),'grad_boost':    tune(GradientBoostingClassifier(random_state=42), {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1]}, x_tr, y_tr),}

    print(f"{'meta model':>15}  {'train acc':>10}  {'test acc':>10}")
    for name, meta in metas.items():
        tr, te = evaluate(build_stacking(meta, x_tr, y_tr), x_tr, y_tr, x_te, y_te)
        print(f"{name:>15}  {tr*100:>9.2f}%  {te*100:>9.2f}%")
