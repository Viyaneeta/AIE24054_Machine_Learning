import numpy as np

def summation(x, w):
    #weighted sum calculation
    return w[0]+np.dot(x,w[1:])

#the main activation functions
def step(y):
    return 1 if y>=0 else 0

def bipolar_step(y):
    if y>0:
        return 1
    elif y==0:
        return 0
    else:
        return -1

def sigmoid(y):
    return 1/(1+np.exp(-y))

def tanh_act(y):
    return np.tanh(y)

def relu(y):
    return max(0,y)

def leaky_relu(y,alpha=0.01):
    return y if y>0 else alpha*y

def activate(y, func='step'):
    funcs={'step': step,'bipolar': bipolar_step,'sigmoid': sigmoid,'tanh': tanh_act,'relu': relu,'leaky_relu': leaky_relu}
    return funcs[func](y)

def comparator(target, output):
    return target-output #calculation for error (actual-pred)

if __name__=='__main__':
    x=np.array([1.0, 2.0])
    w=np.array([0.5, 0.3, 0.7])   #w[0]=bias
    y_in=summation(x, w)
 
    print(f"summation: {y_in:.4f}")
    print(f"comparator: target=1, output=0, error={comparator(1, 0)}")
    print(f"step: {step(y_in)}")
    print(f"bipolar: {bipolar_step(y_in)}")
    print(f"sigmoid: {sigmoid(y_in):.4f}")
    print(f"tanh: {tanh_act(y_in):.4f}")
    print(f"relu: {relu(y_in):.4f}")
    print(f"leaky_relu: {leaky_relu(y_in):.4f}")
