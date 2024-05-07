"""
Created on Thu May 26 22:28:43 2022

@author: pipebabybear
"""

import pennylane as qml
from pennylane import numpy as np
import pickle
import sys

qubit = int(sys.argv[1]) #[8,10,12]
layer_tot = int(sys.argv[2]) #[2,4] #,6,8,10,12,14,16,18,20,25,30] #,35,40] #[2, 4, 8, 16, 25, 50, 75, 100, 150, 200]#[2, 4, 8, 16, 32, 64, 128, 256]
name_data = str(sys.argv[3]) #['random','mnist'] 
sample = int(sys.argv[4]) #[4,40,60]
batch = int(sys.argv[5]) # depending on sample: 0-1 for 4, 0-7 for 40, 0-17 for 60
noise = float(sys.argv[6]) # [0.01,0.025,0.05,0.075,0.10]

name_embed = 'HEE'

###########################################################

pairsize = int(sample*(sample-1)/2)
num_pair = int(sample*(sample-1)/2)

if sample == 4:
    datasize = 2

elif sample == 40:
    datasize = 100

elif sample == 60:
    datasize = 100    
    
batchsize = int(np.ceil(pairsize/datasize))

startidx = batch*datasize
stopidx = (batch+1)*datasize

###########################################################

print('qubit =',qubit)
print('layers =', layer_tot)
print('data type =', name_data)
print('sample =', sample)
print('batch =', batch)
print('datasize =', datasize)

###########################################################

qubit_tot = [2,4,6,8,10,12,14,16,18]
idx_qubit = 0

for i in range(len(qubit_tot)):
    if qubit == qubit_tot[i]:
        idx_qubit = i
        
##############3######33####################################
### load input

if sample == 4:
    filename = 'pairs_ns4'
    infile = open(filename,'rb')
    pairs_ns = pickle.load(infile)
    infile.close()

elif sample == 40:
    filename = 'pairs_ns40'
    infile = open(filename,'rb')
    pairs_ns = pickle.load(infile)
    infile.close()

elif sample == 60:
    filename = 'pairs_ns60'
    infile = open(filename,'rb')
    pairs_ns = pickle.load(infile)
    infile.close()

if name_data == 'random':
    xtrain = pairs_ns[0][idx_qubit][startidx:stopidx]
    
elif name_data == 'mnist':
    xtrain = pairs_ns[1][idx_qubit][startidx:stopidx]

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
    kernel_val = np.zeros(N)
    #k = 0
    for i in range(N):
        kernel_val[i] = kernel_(X[i][0],X[i][1])
        
    return kernel_val

###############################################################################
###############################################################################

ns = len(xtrain)
kernel_tot = np.zeros(ns)

noise_save = noise*100
for i in range(ns):
    print('--- idx data =', i ,' ---')
    kernel_tot[i] = kernel(xtrain[i][0], xtrain[i][1], layer_tot, qubit, noise)

    np.save('fqk_n%i_layer%i_%sdata%i_batch%i_size%i_noise%i.npy'
            %(qubit,layer_tot,name_data,sample,batch,datasize,noise_save),kernel_tot)

######################







