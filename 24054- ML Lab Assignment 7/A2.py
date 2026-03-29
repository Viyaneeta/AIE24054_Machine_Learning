import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# load data
df = pd.read_csv("DCT_mal (2) 1.csv")

x = df.drop("LABEL", axis=1)
y = df["LABEL"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# model
model = SVC()

param_dist = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"]
}

rand_search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=5,
    cv=3,
    scoring="accuracy",
    random_state=42
)

rand_search.fit(x_train, y_train)

best_model = rand_search.best_estimator_

y_pred = best_model.predict(x_test)

print("best parameters:", rand_search.best_params_)
print("accuracy:", accuracy_score(y_test, y_pred))
