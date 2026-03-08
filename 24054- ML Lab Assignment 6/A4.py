import numpy as np
import pandas as pd

def equal_width(data, bins=4):
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val - min_val == 0:
        return np.zeros_like(data)
    width = (max_val - min_val) / bins
    binned = np.floor((data - min_val) / width)
    binned[binned == bins] = bins - 1
    return binned.astype(int)


def equal_frequency(data, bins=4):

    return pd.qcut(data, bins, labels=False, duplicates="drop").to_numpy()


def bin_features(x, bins=4, method="width"):

    xb = np.zeros_like(x)
    for i in range(x.shape[1]):
        if method == "width":
            xb[:, i] = equal_width(x[:, i], bins)
        elif method == "frequency":
            xb[:, i] = equal_frequency(x[:, i], bins)
    return xb


def main():

    data = pd.read_csv("DCT_mal (2) 1.csv")
    x = data.drop("LABEL", axis=1).values
    xb = bin_features(x, 4, "width")
    print("binned data shape:", xb.shape)

if __name__ == "__main__":
    main()
