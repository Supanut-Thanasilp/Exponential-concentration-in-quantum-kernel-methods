"""

Figure 6a: Expressivity-induced concentration for the fidelity kernel


"""

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from data_generator import get_dataset


qubit_tot = [2,4,6,8,10,12]
sample = 60 
layer_tot = [2, 4, 8, 10, 20, 50, 70, 100] 

name_data = 'mnist'
name_embed = 'HEE'
layer_embed = 2


###############################################################################
### Constructing kernels

def layer_circuit(x_in, wires):
    """Building block of the embedding ansatz"""
    
    for i in range(len(wires)):
        qml.RX(x_in[i], wires=i)
    
    qml.broadcast(qml.CNOT, wires=wires, pattern="ring")
    
def ansatz(x, layers, wires):
    """The embedding ansatz"""
    for i in range(layers):
        layer_circuit(x, wires)
        
adjoint_ansatz = qml.adjoint(ansatz)


def kernel(x1, x2, layers, num_qubit):
    dev = qml.device("default.qubit", wires=num_qubit, shots=None)
    wires = dev.wires.tolist()
    
    @qml.qnode(dev)
    def kernel_circuit(x1, x2, layers):
        ansatz(x1, layers, wires=wires)
        adjoint_ansatz(x2, layers, wires=wires)
        return qml.probs(wires=wires)
    
    return kernel_circuit(x1, x2, layers)[0]

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


num_pair = int(sample*(sample-1)/2)
kernel_tot = np.zeros((len(qubit_tot),len(layer_tot),num_pair))


for i in range(len(qubit_tot)):
    print('---- qubits = %i----'%qubit_tot[i])
    
    xtrain, xtest, ytrain, ytest = get_dataset(name_data, qubit_tot[i], sample, 20)
    
    
    for j in range(len(layer_tot)):
        print('layers = %i'%layer_tot[j])
        init_kernel = lambda x1, x2: kernel(x1,x2, layer_tot[j], qubit_tot[i])
        kernel_tot[i][j] = kernel_matrix(xtrain, init_kernel)

varr_ker = np.var(kernel_tot,axis=2)



for i in range(len(qubit_tot)):
    plt.semilogy(layer_tot, varr_ker[i,:],'-o',label=qubit_tot[i])
plt.legend()
plt.show()


for i in range(len(layer_tot)):
    plt.semilogy(qubit_tot, varr_ker[:,i],'-s')
plt.show()

np.save('kernel_%s_%sdata_%isample.npy'%(name_embed,name_data,sample),kernel_tot)






