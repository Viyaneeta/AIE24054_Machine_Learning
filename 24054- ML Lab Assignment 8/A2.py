import numpy as np
import matplotlib.pyplot as plt

def summation(x,w):
    return w[0]+np.dot(x,w[1:])

def step(y):
    return 1 if y>=0 else 0

def comparator(target, output):
    return target-output

def train_perceptron(x_train, y_train, w0=10, w1=0.2, w2=-0.75,lr=0.05, max_epochs=1000, tol=0.002):
    w=np.array([w0, w1, w2],dtype=float)
    sse_history=[]

    for epoch in range(max_epochs):
        sse=0
        for xi, ti in zip(x_train,y_train):
            y_in= summation(xi, w)
            out = step(y_in)
            err = comparator(ti, out)
            w[0]+= lr * err           #updation of bias
            w[1:]+= lr * err * xi      #weight update
            sse+= err ** 2

        sse_history.append(sse)
        if sse <= tol:
            return w, sse_history, epoch + 1

    return w,sse_history,max_epochs


if __name__=='__main__':
    x_and=np.array([[0,0],[0,1],[1,0],[1,1]],dtype=float)
    y_and=np.array([0, 0, 0, 1],dtype=float)

    w_final,sse_hist,epochs=train_perceptron(x_and, y_and)

    print(f"epochs to converge: {epochs}")
    print(f"final weights: bias={w_final[0]:.4f}, w1={w_final[1]:.4f}, w2={w_final[2]:.4f}")

    # predictions
    print("predictions:")
    for xi,ti in zip(x_and,y_and):
        out=step(summation(xi,w_final))
        print(f"input={xi},target={int(ti)},predicted={out}")

    plt.figure()
    plt.plot(sse_hist)
    plt.xlabel("epoch")
    plt.ylabel("sse")
    plt.title("A2: AND gate – step activation convergence")
    plt.tight_layout()
    plt.savefig("a2_and_step.png")
    plt.show()
