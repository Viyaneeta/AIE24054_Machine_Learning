import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

def decision_boundary(x, y):
    x2 = x[:, :2]
    model = DecisionTreeClassifier(max_depth=5)
    model.fit(x2, y)

    x_min, x_max = x2[:,0].min()-1, x2[:,0].max()+1
    y_min, y_max = x2[:,1].min()-1, x2[:,1].max()+1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, alpha=0.3)
    plt.scatter(x2[:,0], x2[:,1], c=y)
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.title("decision boundary")
    plt.show()


def main():

    data = pd.read_csv("DCT_mal (2) 1.csv")
    x = data.drop("LABEL", axis=1).values
    y = data["LABEL"].values
    decision_boundary(x, y)


if __name__ == "__main__":
    main()
