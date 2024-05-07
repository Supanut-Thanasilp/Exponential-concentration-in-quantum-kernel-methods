"""
Created on Thu May 26 22:28:43 2022

@author: pipebabybear
"""

import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
import sympy
import pennylane as qml
from pennylane import numpy as np

import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit

from data_generator import get_dataset


qubit_tot = [2,4,6,8,10,12,14]
sample = 40 #25
layer_tot = [2,4,6,8,10,12,14,20,30,50] #[2, 4, 8, 16, 25, 50, 75, 100, 150, 200]#[2, 4, 8, 16, 32, 64, 128, 256]

noise = [0.001,0.01]
#n_train = 100
#n_test = 20
name_data = 'mnist'
name_embed = 'HEE'
layer_embed = 2



###############################################################################
###############################################################################
### Constructing kernels

def layer_circuit(x_in, wires, p):
    """Building block of the embedding ansatz"""
    
    for i in range(len(wires)):
        qml.RX(x_in[i], wires=i)
        qml.DepolarizingChannel(p, wires=i)
    
    qml.broadcast(qml.CNOT, wires=wires, pattern="ring")


def ansatz(x, layers, wires, p):
    """The embedding ansatz"""
    for i in range(layers):
        layer_circuit(x, wires, p)

###########################################################

def adjoint_layer_circuit(x_in, wires, p):
    """Building reverse block of the embedding ansatz"""
    
    qml.adjoint(qml.broadcast)(qml.CNOT, wires=wires, pattern="ring")
    
    for i in range(len(wires)):
        qml.adjoint(qml.RX)(x_in[i], wires = i)
        qml.DepolarizingChannel(p, wires=i)

def adjoint_ansatz(x_in, layers, wires, p):
    for i in range(layers):
        adjoint_layer_circuit(x_in, wires, p)


#adjoint_ansatz = qml.adjoint(ansatz)

def kernel(x1, x2, layers, num_qubit, p):
    dev = qml.device("default.mixed", wires=num_qubit, shots=None)
    wires = dev.wires.tolist()
    
    @qml.qnode(dev)
    def kernel_circuit(x1, x2, layers, p):
        ansatz(x1, layers, wires=wires, p=p)
        adjoint_ansatz(x2, layers, wires=wires, p=p)
        return qml.probs(wires=wires)
    
    return kernel_circuit(x1, x2, layers, p)[0]

def kernel_matrix(X, kernel_):
    N = len(X)
    kernel_val = np.zeros(int(N*(N-1)/2))
    k = 0
    for i in range(N):
        for j in range(i,N):
            if i == j:
                pass
            else: 
                kernel_val[k] = kernel_(X[i],X[j])
                k += 1
    return kernel_val

###############################################################################
###############################################################################

num_pair = int(sample*(sample-1)/2)
kernel_tot = np.zeros((len(noise),len(qubit_tot),len(layer_tot),num_pair))


for i in range(len(qubit_tot)):
    print('---- qubits = %i----'%qubit_tot[i])
    
    xtrain, xtest, ytrain, ytest = get_dataset(name_data, qubit_tot[i], sample, 20)
    
    
    for j in range(len(layer_tot)):
        print('layers = %i'%layer_tot[j])
        for k in range(len(noise)):
            init_kernel = lambda x1, x2: kernel(x1,x2, layer_tot[j], qubit_tot[i],noise[k])
            kernel_tot[k][i][j] = kernel_matrix(xtrain, init_kernel)
            
    np.save('kernel_noise_test.npy',kernel_tot)



######################3

