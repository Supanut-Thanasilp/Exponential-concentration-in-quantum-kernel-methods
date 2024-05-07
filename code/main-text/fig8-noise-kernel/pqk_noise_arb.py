"""
Created on Thu May 26 22:28:43 2022

@author: pipebabybear
"""

import pennylane as qml
from pennylane import numpy as np
from numpy import linalg as LA
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


def rho_circ(x, layers, num_qubit,p):
    dev = qml.device("default.mixed", wires=num_qubit, shots=None)
    wires = dev.wires.tolist()
    
    @qml.qnode(dev)
    def circ(x, layers,p):
        ansatz(x, layers, wires=wires,p=p)
        return qml.density_matrix(wires)
    
    return circ(x,layers,p)



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


#######################################################
###############################################################################

def proj_kernel(x1,x2,layers, num_qubit, p):
    
    #gamma = 1
    
    rho1 = rho_circ(x1, layers, num_qubit, p)
    rho2 = rho_circ(x2, layers, num_qubit, p)
    
    component = 0
    
    for i in range(num_qubit):
        reduced_rho1 = partial_trace(rho1,[i])
        reduced_rho2 = partial_trace(rho2,[i])
        component += f_norm(reduced_rho1, reduced_rho2)
    
    return component
    


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
comp_tot = np.zeros(ns)

noise_save = noise*100
for i in range(ns):
    print('--- idx data =', i ,' ---')
    comp_tot[i] = proj_kernel(xtrain[i][0], xtrain[i][1], layer_tot, qubit, noise)

    np.save('fqk_n%i_layer%i_%sdata%i_batch%i_size%i_noise%i.npy'
            %(qubit,layer_tot,name_data,sample,batch,datasize,noise_save),comp_tot)





# gamma = 1
# pqk_tot = np.exp(-1*gamma*kernel_tot)

# con_pqk = np.abs(pqk_tot - 1)
# avg_con = np.mean(con_pqk, axis=3)

# import matplotlib.pyplot as plt

# for i in range(len(noise)):
#     plt.semilogy(layer_tot, avg_con[i,0],'-o')

###############################################################################
###############################################################################
###############################################################################

# con = np.zeros((len(noise),len(qubit_tot),len(layer_tot)))

# mix = np.zeros((len(qubit_tot),len(layer_tot),num_pair))
# for i in range(len(qubit_tot)):
#     mix[i] = mix[i] + 1/(2**qubit_tot[i])
    

# con_ker = np.copy(kernel_tot)
# for i in range(len(noise)):
#     con_ker[i] = np.abs(kernel_tot[i] - mix)

# #####################################################3
# knoise = np.mean(kernel_tot, axis=3)

# for i in range(len(noise)):
#     for j in range(len(qubit_tot)):
#         con[i][j] = knoise[i][j] - mix[j]
# ######################################################

# con = np.mean(con_ker, axis=3)

# #####################

# qq = 1 - np.array(noise)

# for i in range(len(noise)):
#     plt.semilogy(layer_tot[:-1], con[i,0][:-1], '-o',label=qq[i])
# plt.legend()












