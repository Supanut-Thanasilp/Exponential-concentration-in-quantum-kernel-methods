"""
Created on Thu May 26 22:28:43 2022

@author: pipebabybear
"""

import pennylane as qml
from pennylane import numpy as np
import pickle
import matplotlib.pyplot as plt

#path = 'fqk/'
path = 'pqk/'

qubit = 8
sample = 40 #25
layer_tot = [2,4,6,8,10,12,14,16,18,20,25] 
noise = [0.01,0.025,0.05,0.075,0.10]
noise_lab = [1, 2,5,7,10]
#n_train = 100
#n_test = 20
name_data = 'mnist'
name_embed = 'HEE_run'

ndata = int(sample*(sample-1)/2) - 100

datasize = 100
batchsize = int(np.ceil(ndata/datasize))

#startidx = batch*datasize
#stopidx = (batch+1)*datasize


batch = [1,2,3,4,5,6,7]

kernel = np.zeros((len(noise),len(layer_tot),ndata))


for i in range(len(noise)):
    for j in range(len(layer_tot)):
        for k in range(len(batch)):
            kernel[i][j][k*datasize:(k+1)*datasize] = np.load(
                path+'fqk_n%i_layer%i_mnistdata%i_batch%i_size%i_noise%i.npy'%(qubit,layer_tot[j],sample,batch[k],datasize,noise_lab[i]))

gamma = 1
pqk = np.exp(-1*gamma*kernel)

mix = 1

con = np.abs(pqk-mix)

avg_con = np.mean(con,axis=2)

for i in range(len(noise)):
    plt.semilogy(layer_tot,avg_con[i],'-o')
    
np.save('pqk_n%i_noise.npy'%qubit, avg_con)

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












