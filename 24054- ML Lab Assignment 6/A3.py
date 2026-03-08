import numpy as np
import pandas as pd

def entropy(y):

    values, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    ent = 0
    for p in probs:
        ent += -p * np.log2(p)
    return ent


def information_gain(x, y, feature):

    total_entropy = entropy(y)
    values, counts = np.unique(x[:, feature], return_counts=True)
    weighted_entropy = 0
    for v, c in zip(values, counts):
        subset = y[x[:, feature] == v]
        weighted_entropy += (c / len(y)) * entropy(subset)
    ig = total_entropy - weighted_entropy

    return ig


def best_root_feature(x, y):
    num_features = x.shape[1]
    gains = []
    for f in range(num_features):
        ig = information_gain(x, y, f)
        gains.append(ig)
    best = np.argmax(gains)
    return best, gains[best]


def main():

    data = pd.read_csv("DCT_mal (2) 1.csv")

    x = data.drop("LABEL", axis=1).values
    y = data["LABEL"].values

    feature, gain = best_root_feature(x, y)

    print("best root feature:", feature)
    print("information gain:", gain)


if __name__ == "__main__":
    main()
