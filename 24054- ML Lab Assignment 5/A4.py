#k means clustering on the dataset
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

df = pd.read_csv("DCT_mal (2) 1.csv")
X = df.drop(columns=["LABEL"])   #removing the target variable
y = df["LABEL"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def perform_kmeans(X_train, k):
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X_train)
    return kmeans

kmeans = perform_kmeans(X_train, k=2)

print("Cluster Labels:", kmeans.labels_)
print("Cluster Centers:", kmeans.cluster_centers_)
