"""

Figure 9: Exponential concentration in kernel target alignment.


"""

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from data_generator import get_dataset

###############################################################################

qubit_tot = [2,6,10,12,14,16,18] #[10,12,14,16,18] 
sample = 10 #8 #10 #26 #40 
sample_params = 500 #400
layer_tot = [1] #[2,35,70] #,128]

###############################################################################
#n_train = 100
#n_test = 20
name_data = 'random'#'fashion-mnist'
name_embed = 'HEE'

###############################################################################
################ Kernel for a give pair
########################################################

def param_circuit(params, wires):
    for i in range(len(wires)):
        qml.RY(params[i], wires = i)
    
def layer_circuit(x_in, wires):
    """Building block of the embedding ansatz"""
    
    for i in range(len(wires)):
        qml.RX(x_in[i], wires=i)
    
    qml.broadcast(qml.CNOT, wires=wires, pattern="ring")
    
def param_ansatz(x, params, layers, wires):
    """Parametrzed embedding ansatz"""
    
    param_circuit(params, wires)
    
    for i in range(layers):
        layer_circuit(x, wires)
        
adjoint_ansatz = qml.adjoint(param_ansatz)


def kernel(x1, x2, params, layers, num_qubit):
    dev = qml.device("default.qubit", wires=num_qubit, shots=None)
    wires = dev.wires.tolist()
    
    @qml.qnode(dev)
    def kernel_circuit(x1, x2, params, layers):
        param_ansatz(x1, params, layers, wires=wires)
        adjoint_ansatz(x2, params, layers, wires=wires)
        return qml.probs(wires=wires)
    
    return kernel_circuit(x1, x2, params, layers)[0]

###############################################################################

def ta_term(x1, x2, y1, y2, params, layers, num_qubit):
    kerr = kernel(x1, x2, params, layers, num_qubit)
    numerator = y1*y2*kerr
    denominator1 = np.square(kerr)
    denominator2 = np.square(y1*y2)
    
    return numerator, denominator1, denominator2



def TA(X,Y, params, layers, num_qubit):
    N = len(X)
    numer = 0
    denom1 = 0
    denom2 = 0
    for i in range(N):
        for j in range(i,N):
            if i == j:
                numer += Y[i]*Y[j]
                denom1 += 1
                denom2 += np.square(Y[i]*Y[j])
            else:
                n1, d1, d2 = ta_term(X[i],X[j],Y[i],Y[j],params, layers, num_qubit)
                numer += 2*n1
                denom1 += 2*d1
                denom2 += 2*d2
    return numer/np.sqrt(denom1*denom2), numer, denom1, denom2


###############################################################################

ta = np.zeros((len(qubit_tot),len(layer_tot),sample_params))
numer = np.zeros((len(qubit_tot),len(layer_tot),sample_params))
denom1 = np.zeros((len(qubit_tot),len(layer_tot),sample_params))
denom2 = np.zeros((len(qubit_tot),len(layer_tot),sample_params))

import time
t1 = time.time()
for i in range(len(qubit_tot)):
    print('---- qubits = %i----'%qubit_tot[i])
    
    xtrain, xtest, ytrain, ytest = get_dataset(name_data, qubit_tot[i], sample, 20)
    
    for j in range(len(layer_tot)):
        print('layers = %i'%layer_tot[j])
        
        for k in range(sample_params):
            if k%10 == 0:
                print(k)
            params = np.random.uniform(-2*np.pi, 2*np.pi, qubit_tot[i])
            ta[i,j,k], numer[i,j,k], denom1[i,j,k], denom2[i,j,k] = TA(xtrain, ytrain, params, layer_tot[j],qubit_tot[i])
    
    np.save('ta_test_%s_%s.npy'%(name_data,name_embed),ta)
    np.save('numer_test_%s_%s.npy'%(name_data,name_embed),numer)
    np.save('denom1_test_%s_%s.npy'%(name_data,name_embed),denom1)
    np.save('denom2_test_%s_%s.npy'%(name_data,name_embed),denom2)
  
t2 = time.time()
print(t2-t1)

var_ta = np.var(ta, axis=2)
var_numer = np.var(numer,axis=2)
var_denom = np.var(denom1, axis=2)

for i in range(len(qubit_tot)):
    plt.semilogy(layer_tot, var_numer[i,:],'-o',label=qubit_tot[i])
plt.legend()
plt.show()

for i in range(len(qubit_tot)):
    plt.semilogy(layer_tot, var_denom[i,:],'-o',label=qubit_tot[i])
plt.legend()
plt.show()


for i in range(len(qubit_tot)):
    plt.semilogy(layer_tot, var_ta[i,:],'-o',label=qubit_tot[i])
plt.legend()
plt.show()

#plt.semilogy(qubit_tot, var_numer,'-o')
#plt.semilogy(qubit_tot, var_denom,'-o')
plt.semilogy(qubit_tot, var_ta,'-o')




