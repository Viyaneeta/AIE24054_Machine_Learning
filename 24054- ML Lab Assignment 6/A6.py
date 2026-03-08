import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

def draw_tree(model, feature_names):
    plt.figure(figsize=(20,10))
    plot_tree(
        model,
        feature_names=feature_names,
        filled=True,
        rounded=True
    )
    plt.title("decision tree for palm leaf classification")
    plt.show()


def main():

    data = pd.read_csv("DCT_mal (2) 1.csv")
    x = data.drop("LABEL", axis=1).values
    y = data["LABEL"].values
    model = DecisionTreeClassifier(criterion="entropy")
    model.fit(x, y)
    draw_tree(model, data.columns[:-1])

if __name__ == "__main__":
    main()
