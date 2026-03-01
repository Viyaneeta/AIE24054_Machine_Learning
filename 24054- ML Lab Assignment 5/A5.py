#calculating silhouette score, db index and ch score
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

df = pd.read_csv("DCT_mal (2) 1.csv")
X = df.drop(columns=["LABEL"])
y = df["LABEL"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def clustering_scores(X_train, k):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X_train)
    sil = silhouette_score(X_train, kmeans.labels_)
    ch  = calinski_harabasz_score(X_train, kmeans.labels_)
    db  = davies_bouldin_score(X_train, kmeans.labels_)
    return sil, ch, db

sil, ch, db = clustering_scores(X_train, k=2)

print(f"Silhouette Score:{sil:.4f}")
print(f"CH Score:{ch:.4f}")
print(f"DB Index:{db:.4f}")
