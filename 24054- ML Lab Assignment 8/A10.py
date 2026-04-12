import numpy as np

def sm(x, w):
    return np.dot(x, w)

def act(y):
    return 1 if y >= 0 else 0

def train(x, t, w, lr, ep=1000):
    for e in range(ep):
        for i in range(len(x)):
            y = sm(x[i], w)
            o = np.array([act(y[0]), act(y[1])])
            d = t[i] - o
            w = w + lr * np.outer(x[i], d)
    return w

#input
x = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
#one hot target
t = np.array([[1,0],[1,0],[1,0],[0,1]])
#weights
w = np.random.rand(3,2)
lr = 0.05
w = train(x, t, w, lr)
print(w)
