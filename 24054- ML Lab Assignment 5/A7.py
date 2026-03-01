#using elbow plot to determine the best k value
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "DCT_mal (2) 1.csv"))
X = df.drop(columns=["LABEL"])
y = df["LABEL"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

distortions=[]
#fitting k means for k=2
for k in range(2, 20):
    kmeans = KMeans(n_clusters=k).fit(X_train)
    distortions.append(kmeans.inertia_)
#elbow point is where the drop in inertia slows down the most
diffs = [distortions[i-1] - distortions[i] for i in range(1, len(distortions))]
elbow_k = list(range(2, 20))[diffs.index(max(diffs))]

#intertia decreases as k increses
plt.plot(range(2, 20), distortions, marker="o")
plt.title("Elbow Plot")
plt.xlabel("k")
plt.ylabel("Inertia (Distortion)")
plt.show()

print(f"Optimal k (elbow point): {elbow_k}")
