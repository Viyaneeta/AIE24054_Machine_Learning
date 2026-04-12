import numpy as np
import matplotlib.pyplot as plt

def summation(x,w):
    return w[0]+np.dot(x,w[1:])

def step(y):
    return 1 if y>=0 else 0

def bipolar_step(y):
    if y>0:    return 1
    elif y==0: return 0
    else:
        return -1

def sigmoid(y):
    return 1/(1+np.exp(-y))

def relu(y):
    return max(0,y)

def activate(y, func):
    return {'step': step,'bipolar': bipolar_step,'sigmoid': sigmoid,'relu': relu}[func](y)

def comparator(target, output):
    return target-output

def train_perceptron(x_train,y_train,func='step',w0=10,w1=0.2,w2=-0.75,lr=0.05, max_epochs=1000, tol=0.002):
    w=np.array([w0, w1, w2],dtype=float)
    sse_history=[]

    for epoch in range(max_epochs):
        sse=0
        for xi,ti in zip(x_train, y_train):
            y_in =summation(xi, w)
            out= activate(y_in, func)
            err= comparator(ti, out)
            w[0]+= lr*err
            w[1:]+= lr*err*xi
            sse+= err**2
        sse_history.append(sse)
        if sse<=tol:
            return sse_history,epoch+1

    return sse_history,max_epochs


if __name__=='__main__':
    x_xor=np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y_xor=np.array([0, 1, 1, 0], dtype=float)

    funcs=['step','bipolar','sigmoid','relu']

    plt.figure()
    print(f"{'activation':12s}{'epochs':>8} {'converged':>10}")
    for fn in funcs:
        sse_hist,epochs=train_perceptron(x_xor, y_xor, func=fn)
        converged=epochs<1000
        print(f"{fn:12s}{epochs:>8} {str(converged):>10}")
        plt.plot(sse_hist,label=fn)

    plt.xlabel("epoch")
    plt.ylabel("sse")
    plt.title("A5: XOR gate – all activations (expected: no convergence)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("a5_xor.png")
    plt.show()
