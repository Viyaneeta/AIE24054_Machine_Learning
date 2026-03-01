#k means clustering for different values of k
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

df= pd.read_csv("DCT_mal (2) 1.csv")
X= df.drop(columns=["LABEL"])
y= df["LABEL"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def clustering_scores(X_train, k):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X_train)   #range is -1 to 1
    sil= silhouette_score(X_train, kmeans.labels_)          #higher value is better
    ch= calinski_harabasz_score(X_train, kmeans.labels_)       #lower value is better
    db= davies_bouldin_score(X_train, kmeans.labels_) 
    return sil, ch, db

sil_scores=[]
ch_scores=[]
db_scores=[]
k_range=range(2,20)

for k in k_range:
    sil, ch, db =clustering_scores(X_train, k)
    sil_scores.append(sil)
    ch_scores.append(ch)
    db_scores.append(db)
    print(f"k={k}  Silhouette={sil:.4f}  CH={ch:.4f}  DB={db:.4f}")

plt.figure()
plt.plot(k_range, sil_scores, marker="o")
plt.title("silhouette Score vs k")
plt.xlabel("k")
plt.ylabel("silhouette score")
plt.show()

plt.figure()
plt.plot(k_range, ch_scores, marker="o")
plt.title("calinski-harabasz score vs k")
plt.xlabel("k")
plt.ylabel("ch Score")
plt.show()

plt.figure()
plt.plot(k_range, db_scores, marker="o")
plt.title("davies-bouldin index vs k")
plt.xlabel("k")
plt.ylabel("db Index")
plt.show()
