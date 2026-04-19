#a2
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def load(path):
    df = pd.read_csv("DCT_mal (2) 1")
    return df.drop('LABEL', axis=1).values, df['LABEL'].values

def build_pipeline(clf):
    return Pipeline([('scaler', StandardScaler()),('pca',    PCA(n_components=50)),('clf',    clf), ])

def tune_pipeline(pipe, params, x, y):
    search = RandomizedSearchCV(pipe, params, n_iter=10, cv=3, n_jobs=-1, random_state=42)
    search.fit(x, y)
    print(f"  best params: {search.best_params_}")
    return search.best_estimator_


if __name__ == '__main__':
    x, y = load("DCT_mal (2) 1")
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    clfs = {'random_forest': ( build_pipeline(RandomForestClassifier(random_state=42)), {'pca__n_components': [30, 50, 80], 'clf__n_estimators': [50, 100], 'clf__max_depth': [None, 10, 20]} ), 'svc': (build_pipeline(SVC(kernel='rbf', random_state=42)),{'pca__n_components': [30, 50, 80], 'clf__C': [0.1, 1, 10], 'clf__gamma': ['scale', 'auto']}),'grad_boost': ( build_pipeline(GradientBoostingClassifier(random_state=42)), {'pca__n_components': [30, 50, 80], 'clf__n_estimators': [50, 100], 'clf__learning_rate': [0.05, 0.1]}),}

    print(f"{'classifier':>15}  {'train acc':>10}  {'test acc':>10}  {'cv mean':>10}")
    for name, (pipe, params) in clfs.items():
        print(f"\ntuning {name}")
        best = tune_pipeline(pipe, params, x_tr, y_tr)
        cv = cross_val_score(best, x_tr, y_tr, cv=3, n_jobs=-1).mean()
        print(f"{name:>15}  {accuracy_score(y_tr, best.predict(x_tr))*100:>9.2f}%  {accuracy_score(y_te, best.predict(x_te))*100:>9.2f}%  {cv*100:>9.2f}%")
