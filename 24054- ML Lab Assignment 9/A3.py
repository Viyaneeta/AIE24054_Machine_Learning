#A3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from lime.lime_tabular import LimeTabularExplainer


def load(path):
    df = pd.read_csv("DCT_mal (2) 1")
    return df.drop('LABEL', axis=1).values, df['LABEL'].values, df.drop('LABEL', axis=1).columns.tolist()

def build_pipeline(x_tr, y_tr):
    pipe = Pipeline([('scaler', StandardScaler()),('pca',    PCA(n_components=50)),('clf',    RandomForestClassifier(n_estimators=100, random_state=42)),])
    pipe.fit(x_tr, y_tr)
    return pipe


if __name__ == '__main__':
    x, y, feat_names = load("DCT_mal (2) 1")
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    pipe = build_pipeline(x_tr, y_tr)
    print(f"pipeline test acc: {accuracy_score(y_te, pipe.predict(x_te))*100:.2f}%")

    explainer = LimeTabularExplainer(x_tr,feature_names=feat_names,class_names=[str(c) for c in np.unique(y)],mode='classification')

    for i in range(3):
        exp = explainer.explain_instance(x_te[i], pipe.predict_proba, num_features=10)
        print(f"\nsample {i} – true: {y_te[i]}, predicted: {pipe.predict([x_te[i]])[0]}")
        for feat, weight in exp.as_list():
            print(f"  {feat:>40s}  weight: {weight:+.4f}")
        fig = exp.as_pyplot_figure()
        fig.tight_layout()
        fig.savefig(f"a3_lime_sample_{i}.png")
        plt.close()
