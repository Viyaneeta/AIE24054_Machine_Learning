import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# load data
df = pd.read_csv("DCT_mal (2) 1.csv")

# features and labels
x = df.drop("LABEL", axis=1)
y = df["LABEL"]

# split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42, stratify=y
)

# scale
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# train model
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

# get feature importance
importance = model.feature_importances_

# plot
plt.figure()
plt.plot(importance)
plt.title("Feature Importance of DCT Coefficients")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.show()
