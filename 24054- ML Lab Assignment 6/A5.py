import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def build_tree(x, y):
    model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=5,
        random_state=0
    )
    model.fit(x, y)
    return model


def main():

    data = pd.read_csv("DCT_mal (2) 1.csv")
    x = data.drop("LABEL", axis=1).values
    y = data["LABEL"].values
    model = build_tree(x, y)
    print("decision tree built successfully")


if __name__ == "__main__":
    main()
