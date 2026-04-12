import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def summation(x,w):
    return w[0]+np.dot(x,w[1:])

def sigmoid(y):
    return 1/(1+np.exp(-y))

def comparator(target,output):
    return target-output

def normalize(x):
    return (x-x.mean(axis=0))/x.std(axis=0)

def train_perceptron(x_train,y_train,lr=0.05,max_epochs=1000,tol=0.002):
    rng=np.random.default_rng(42)
    n=x_train.shape[1]
    w=rng.uniform(-0.5, 0.5, n + 1)
    for epoch in range(max_epochs):
        sse=0
        for xi,ti in zip(x_train, y_train):
            y_in= summation(xi, w)
            out =sigmoid(y_in)
            err=comparator(ti, out)
            w[0]+=lr*err
            w[1:]+= lr*err*xi
            sse   +=err**2
        if sse<=tol:
            return w
    return w

def predict_perceptron(x, w):
    return np.array([1 if sigmoid(summation(xi, w)) >= 0.5 else 0 for xi in x])
def pseudo_inverse_solution(x, y):
    x_aug=np.hstack([np.ones((len(x), 1)), x])
    return np.linalg.pinv(x_aug) @ y
def predict_pinv(x, w):
    x_aug=np.hstack([np.ones((len(x), 1)), x])
    return (x_aug@w>=0.5).astype(int)


if __name__ == '__main__':

    df=pd.read_csv('DCT_mal (2) 1.csv')
    x_m=df.drop('LABEL', axis=1).values.astype(float)
    y_m=df['LABEL'].values

    #class 0 vs all the other
    y_bin=(y_m==0).astype(float)
    x_tr, x_te, y_tr, y_te = train_test_split(x_m, y_bin, test_size=0.2, random_state=42, stratify=y_bin)
    scaler = StandardScaler()
    x_tr_s = scaler.fit_transform(x_tr)
    x_te_s = scaler.transform(x_te)
    w_perc_m  = train_perceptron(x_tr_s, y_tr)
    pred_p_m  = predict_perceptron(x_te_s, w_perc_m)
    w_pinv_m  = pseudo_inverse_solution(x_tr_s, y_tr)
    pred_pi_m = predict_pinv(x_te_s, w_pinv_m)

    print(f"task: class 0 vs rest (binary, one-vs-rest)")
    print(f"test samples: {len(y_te)}")
    print(f"perceptron accuracy: {np.mean(pred_p_m == y_te)*100:.2f}%")
    print(f"pseudo-inv accuracy: {np.mean(pred_pi_m == y_te)*100:.2f}%")
