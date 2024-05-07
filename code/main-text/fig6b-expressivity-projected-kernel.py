"""

Figure 6b: Expressivity-induced concentration for the projected kernel


"""

import pennylane as qml
from pennylane import numpy as np

import matplotlib.pyplot as plt

from data_generator import get_dataset
from numpy import linalg as LA


qubit_tot = [2,4,6,8,10] #,12]
sample = 40
layer_tot = [2, 4, 8,10,15,20,25,30,35,40,45,50] #,70,100,120] 

name_data = 'mnist'
name_embed = 'HEE-invest-test'


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

def rho_circ(x, layers, num_qubit):
    dev = qml.device("default.qubit", wires=num_qubit, shots=None)
    wires = dev.wires.tolist()
    
    @qml.qnode(dev)
    def circ(x, layers):
        ansatz(x, layers, wires=wires)
        return qml.density_matrix(wires)
    
    return circ(x,layers)

def partial_trace(rho, qubit_2_keep):
    """ Calculate the partial trace for qubit system
    Parameters
    ----------
    rho: np.ndarray
        Density matrix
    qubit_2_keep: list
        Index of qubit to be kept after taking the trace
    Returns
    -------
    rho_res: np.ndarray
        Density matrix after taking partial trace
    """
    num_qubit = int(np.log2(rho.shape[0]))
    qubit_axis = [(i, num_qubit + i) for i in range(num_qubit)
                  if i not in qubit_2_keep]
    minus_factor = [(i, 2 * i) for i in range(len(qubit_axis))]
    minus_qubit_axis = [(q[0] - m[0], q[1] - m[1])
                        for q, m in zip(qubit_axis, minus_factor)]
    rho_res = np.reshape(rho, [2, 2] * num_qubit)
    qubit_left = num_qubit - len(qubit_axis)
    for i, j in minus_qubit_axis:
        rho_res = np.trace(rho_res, axis1=i, axis2=j)
    if qubit_left > 1:
        rho_res = np.reshape(rho_res, [2 ** qubit_left] * 2)

    return rho_res

def f_norm(A,B):
    diff = A - B
    return np.square(LA.norm(diff, ord='fro'))


###############################################################################

def proj_kernel(x1,x2,layers, num_qubit):
    
    #gamma = 1
    
    rho1 = rho_circ(x1, layers, num_qubit)
    rho2 = rho_circ(x2, layers, num_qubit)
    
    component = 0
    
    for i in range(num_qubit):
        reduced_rho1 = partial_trace(rho1,[i])
        reduced_rho2 = partial_trace(rho2,[i])
        component += f_norm(reduced_rho1, reduced_rho2)
    
    return component
    
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

num_pair = int(sample*(sample-1)/2)
component_tot = np.zeros((len(qubit_tot),len(layer_tot),num_pair))

for i in range(len(qubit_tot)):
    print('---- qubits = %i----'%qubit_tot[i])
    
    xtrain, xtest, ytrain, ytest = get_dataset(name_data, qubit_tot[i], sample, 20)
    
    
    for j in range(len(layer_tot)):
        print('layers = %i'%layer_tot[j])
        init_kernel = lambda x1, x2: proj_kernel(x1,x2, layer_tot[j], qubit_tot[i])
        component_tot[i][j] = kernel_matrix(xtrain, init_kernel)

    np.save('proj_component_%s_%sdata_%isample.npy'%(name_embed,name_data,sample),component_tot)

gamma = 1
kernel_tot = np.exp(-1*gamma*component_tot)


varr_ker = np.var(kernel_tot,axis=2)
varr_component = np.var(component_tot,axis=2)

for i in range(len(qubit_tot)):
    plt.semilogy(layer_tot, varr_ker[i,:],'-o',label=qubit_tot[i])
plt.legend()
plt.show()


for i in range(len(qubit_tot)):
    plt.semilogy(layer_tot, varr_component[i,:],'-o',label=qubit_tot[i])
plt.legend()
plt.show()











