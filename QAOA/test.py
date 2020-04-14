import numpy as np
import scipy.linalg as sl
import cirq

X = np.matrix([[0, 1], [1, 0]])
theta = 0.7*np.pi

X1 = cirq.XPowGate(exponent = 2 * theta / np.pi).matrix()
X2 = sl.expm(-1.j * theta * X)

print(np.divide(X1, X2))