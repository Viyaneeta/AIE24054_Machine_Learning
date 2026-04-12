import numpy as np
import matplotlib.pyplot as plt

def summation(x, w):
    return w[0]+np.dot(x,w[1:])

def step(y):
    return 1 if y>=0 else 0

def comparator(target,output):
    return target-output

 
def train_perceptron(x_train, y_train, lr, w0=10, w1=0.2, w2=-0.75, max_epochs=1000, tol=0.002):
    w = np.array([w0, w1, w2], dtype=float)
    for epoch in range(max_epochs):
        sse=0
        for xi,ti in zip(x_train, y_train):
            y_in=summation(xi, w)
            out=step(y_in)
            err=comparator(ti, out)
            w[0]+=lr*err
            w[1:]+=lr*err*xi
            sse+=err ** 2
        if sse<=tol:
            return epoch+1
 
    return max_epochs

if __name__=='__main__':
    x_and=np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y_and=np.array([0, 0, 0, 1], dtype=float)

    lrs=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    epochs_list=[]

    print(f"{'lr':>6}{'epochs':>8}")
    for lr in lrs:
        ep=train_perceptron(x_and, y_and, lr=lr)
        epochs_list.append(ep)
        print(f"{lr:>6.1f}{ep:>8}")

    plt.figure()
    plt.plot(lrs, epochs_list, marker='o')
    plt.xlabel("learning rate")
    plt.ylabel("epochs to converge")
    plt.title("learning rate vs convergence epochs (AND, step)")
    plt.tight_layout()
    plt.savefig("a4_lr_vs_epochs.png")
    plt.show()
