import numpy as np
import pandas as pd

def entropy(y):

    values, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    ent = 0
    for p in probs:
        ent += -p * np.log2(p)
    return ent


def main():

    data = pd.read_csv("DCT_mal (2) 1.csv")

    y = data["LABEL"].values

    e = entropy(y)

    print("entropy:", e)


if __name__ == "__main__":
    main()
