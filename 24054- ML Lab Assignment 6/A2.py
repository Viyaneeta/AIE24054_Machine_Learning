import numpy as np
import pandas as pd

def gini(y):

    values, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    g = 1 - np.sum(probs ** 2)
    return g


def main():

    data = pd.read_csv("DCT_mal (2) 1.csv")
    y = data["LABEL"].values
    g = gini(y)
    print("gini index:", g)

if __name__ == "__main__":
    main()
