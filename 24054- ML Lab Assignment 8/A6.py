import numpy as np
def summation(x,w):
    return w[0]+np.dot(x,w[1:])

def sigmoid(y):
    return 1/(1+np.exp(-y))

def comparator(target,output):
    return target-output

def train_perceptron(x_train,y_train,lr=0.05,max_epochs=1000,tol=0.002):
    rng=np.random.default_rng(42)
    n =x_train.shape[1]
    w =rng.uniform(-0.5, 0.5, n + 1)#w[0] is bias
    sse_history=[]

    for epoch in range(max_epochs):
        sse=0
        for xi,ti in zip(x_train, y_train):
            y_in= summation(xi, w)
            out= sigmoid(y_in)
            err= comparator(ti, out)
            w[0]+= lr*err
            w[1:]+= lr*err*xi
            sse+= err**2
        sse_history.append(sse)
        if sse<=tol:
            return w,epoch+1

    return w,max_epochs

def predict(x,w):
    return [1 if sigmoid(summation(xi, w))>=0.5 else 0 for xi in x]

def normalize(x):
    return (x-x.mean(axis=0))/x.std(axis=0)


if __name__=='__main__':
    x =np.array([[20, 6, 2, 386],[16, 3, 6, 289],[27, 6, 2, 393],[19, 1, 2, 110],[24, 4, 2, 280],[22, 1, 5, 167],[15, 4, 2, 271],[18, 4, 2, 274],[21, 1, 4, 148],[16, 2, 4, 198]], dtype=float)
    y =np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0], dtype=float)

    x_norm=normalize(x)
    w_final,epochs =train_perceptron(x_norm, y)

    preds=predict(x_norm, w_final)
    acc=np.mean(np.array(preds)==y) * 100

    print(f"epochs to converge: {epochs}")
    print(f"accuracy: {acc:.1f}%\n")
    print(f"{'customer':>10}{'predicted':>10}{'actual':>8}")
    for i, (p, t) in enumerate(zip(preds, y.astype(int))):
        label = 'high' if p == 1 else 'low'
        print(f"{'C_'+str(i+1):>10} {label:>10}{('high' if t else 'low'):>8}")
