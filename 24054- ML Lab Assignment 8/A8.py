import numpy as np
import matplotlib.pyplot as plt


def sigmoid(y):
    return 1/(1+np.exp(-y))

def train_mlp(x_train, y_train, lr=0.05, max_epochs=1000, tol=0.002):
    rng=np.random.default_rng(0)
    v=rng.uniform(-0.05, 0.05, (2, 2))   
    bv=rng.uniform(-0.05, 0.05, 2)         #hidden bias
    w=rng.uniform(-0.05, 0.05, 2)         
    bw=float(rng.uniform(-0.05, 0.05))     #output bias

    sse_history=[]
    for epoch in range(max_epochs):
        sse=0
        for xi,ti in zip(x_train, y_train):
            h_in=bv+xi@v
            h_out=np.array([sigmoid(h_in[0]),sigmoid(h_in[1])])
            o_in=bw+np.dot(h_out, w)
            o_out=sigmoid(o_in)

            err=ti-o_out
            sse+=err ** 2       #hidden delts
            delta_o=o_out * (1 - o_out) * err   #output delta
            delta_h=np.array([ h_out[j] * (1 - h_out[j]) * delta_o * w[j]for j in range(2)])
            w+=lr * delta_o * h_out
            bw+=lr * delta_o
            for j in range(2):
                v[:, j] += lr * delta_h[j] * xi
                bv[j]   += lr * delta_h[j]

        sse_history.append(sse)
        if sse<=tol:
            return v, bv, w, bw, sse_history, epoch + 1
    return v, bv, w, bw, sse_history, max_epochs

def predict_mlp(x, v, bv, w, bw):
    preds=[]
    for xi in x:
        h_out= np.array([sigmoid(bv[j] + np.dot(xi, v[:, j])) for j in range(2)])
        o_out= sigmoid(bw + np.dot(h_out, w))
        preds.append(1 if o_out >= 0.5 else 0)
    return preds


if __name__ == '__main__':
    x_and = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y_and = np.array([0, 0, 0, 1], dtype=float)
    v, bv, w, bw, sse_hist, epochs = train_mlp(x_and, y_and)
    converged=epochs<1000
    print(f"converged: {converged}")
    print(f"epochs: {epochs}")
    print(f"final sse: {sse_hist[-1]:.5f}")

    preds = predict_mlp(x_and, v, bv, w, bw)
    print("\npredictions:")
    for xi, ti, p in zip(x_and, y_and.astype(int), preds):
        print(f"  input={xi}, target={ti}, predicted={p}")

    plt.figure()
    plt.plot(sse_hist)
    plt.xlabel("epoch")
    plt.ylabel("sse")
    plt.title("A8: MLP backprop – AND gate (sigmoid, lr=0.05)")
    plt.tight_layout()
    plt.savefig("a8_mlp_and.png")
    plt.show()
