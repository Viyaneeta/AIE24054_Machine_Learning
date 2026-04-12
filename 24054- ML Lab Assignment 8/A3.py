import numpy as np
import matplotlib.pyplot as plt

def summation(x, w):
    return w[0]+np.dot(x, w[1:])

def bipolar_step(y):
    if y>0:   return 1
    elif y==0: return 0
    else:
        return -1

def sigmoid(y):
    return 1/(1+np.exp(-y))

def relu(y):
    return max(0,y)

def activate(y,func):
    return {'bipolar':bipolar_step,'sigmoid':sigmoid,'relu':relu}[func](y)

def comparator(target, output):
    return target-output

def train_perceptron(x_train, y_train, func, w0=10, w1=0.2, w2=-0.75,lr=0.05, max_epochs=1000, tol=0.002):
    w = np.array([w0, w1, w2], dtype=float)
    sse_history = []

    for epoch in range(max_epochs):
        sse=0
        for xi, ti in zip(x_train, y_train):
            y_in=summation(xi, w)
            out=activate(y_in, func)
            err=comparator(ti, out)
            w[0]+=lr*err
            w[1:]+=lr*err*xi
            sse+=err**2
        sse_history.append(sse)
        if sse <= tol:
            return sse_history,epoch+1

    return sse_history,max_epochs


if __name__=='__main__':
    x_and=np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y_and=np.array([0, 0, 0, 1], dtype=float)

    funcs=['bipolar', 'sigmoid', 'relu']
    results={}

    plt.figure()
    for fn in funcs:
        sse_hist,epochs=train_perceptron(x_and, y_and, func=fn)
        results[fn]=epochs
        plt.plot(sse_hist,label=fn)
        print(f"{fn:10s}- epochs: {epochs}")

    print("summary:", results)

    plt.xlabel("epoch")
    plt.ylabel("sse")
    plt.title("Activation function comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("a3_activations.png")
    plt.show()
