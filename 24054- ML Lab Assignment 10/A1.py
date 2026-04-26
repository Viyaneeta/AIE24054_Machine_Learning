import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def load_data(path):
    data = pd.read_csv(path)
    x = data.drop(columns=["LABEL"])
    return x

def preprocess(x):
    sc = StandardScaler()
    return sc.fit_transform(x)

def compute_corr(x):
    return pd.DataFrame(x).corr()

def plot_heatmap(corr):
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, cmap='coolwarm')
    plt.title('correlation heatmap')
    plt.show()

# main
path = 'DCT_mal (2) 1.csv'

x = load_data(path)
x_scaled = preprocess(x)

corr = compute_corr(x_scaled)
plot_heatmap(corr)
