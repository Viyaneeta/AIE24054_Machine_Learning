import numpy as np

#summation
def sm(x, w):
    return np.dot(x,w)
#activation
def act(y):
    return 1 if y>=0 else 0

#error
def err(t, o):
    return t-o

#train perceptron
def train(x, t, w, lr, ep=1000):
    for e in range(ep):
        se = 0
        for i in range(len(x)):
            y = sm(x[i], w)
            o = act(y)
            d = err(t[i], o)
            w = w + lr * d * x[i]
            se += d**2
        if se <= 0.002:
            break
    return w

# main
x = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
t = np.array([0,1,1,0])

w = np.array([0.2, -0.75, 10])
lr = 0.05

w = train(x, t, w, lr)
print(w)
