from sklearn.neural_network import MLPClassifier
import numpy as np
#data
x = np.array([[0,0],[0,1],[1,0],[1,1]])
#AND
t1 = np.array([0,0,0,1])
#XOR
t2 = np.array([0,1,1,0])
#model
m = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000)
#AND
m.fit(x, t1)
print(m.predict(x))
#XOR
m.fit(x, t2)
print(m.predict(x))
