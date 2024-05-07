"""

Supplementary Figure 4b: Effect of exponential concentration on estimated Gram matrix (the SWAP test + fidelity kernel)


"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats



#### obtain datasets

def get_random_dataset(dataset_dim, n_train, n_test): # generate random training data
    
    x_train = np.random.uniform(0, 2*np.pi, (n_train,dataset_dim))
    
    x_test = np.random.uniform(0, 2*np.pi, (n_test,dataset_dim))
    return x_train, x_test

def label_generator(x_input):
    return np.mean(x_input,axis=1)




#### define kernel


def fidelity_kernel(xvi, xvj):
    diff = (xvi - xvj)/2
    coss = np.square(np.cos(diff))
    return np.prod(coss)



def kernel_matrix(X, kernel_):
    N = len(X)
    kernel_val = np.zeros((N,N))
    for i in range(N):
        for j in range(i,N):
            if i == j:
                kernel_val[i][i] = 1
            else: 
                kk = kernel_(X[i],X[j])
                kernel_val[i][j] = kk
                kernel_val[j][i] = kk
    return kernel_val


def sk_swap(kerr, n_sample):
    prob_p = 1/2 + kerr/2
    prob_m = 1/2 - kerr/2
    samples = np.random.choice([-1,1], n_sample, p=[prob_m,prob_p])
    return samples



def sk_swap_pval(kerr, n_sample):
    prob_p = 1/2 + kerr/2
    prob_m = 1/2 - kerr/2
    samples = np.random.choice([0,1], n_sample, p=[prob_m,prob_p])
    return samples



def sk_overlap(kerr, n_sample):
    prob_1 = kerr
    prob_0 = 1 - kerr
    samples = np.random.choice([0,1], n_sample, p=[prob_0,prob_1])
    return samples


def kernel_trains(X, kernel_):
    N = len(X)
    kernel_val = np.zeros((N,N))
    kernel_val = np.zeros((int(N*(N-1)/2)))
    k = 0
    for i in range(N):
        for j in range(i,N):
            if i != j:
                kernel_val[k] = kernel_(X[i],X[j])
                k += 1
                
    return kernel_val


def kernel_matrix_est(X, kernel_, n_shot, m_test):
    N = len(X)
    kernel_val = np.zeros((N,N))
    #kernel_est = np.zeros((N,N))
    shot_outcomes = np.zeros((int(N*(N-1)/2),n_shot))
    k = 0
    for i in range(N):
        for j in range(i,N):
            if i == j:
                kernel_val[i][i] = 1
                # kernel_est[i][i] = 1
            else: 
                kk = kernel_(X[i],X[j])
                kernel_val[i][j] = kk
                kernel_val[j][i] = kk
                
                if m_test == 'overlap':
                    shot_outcomes[k] = sk_overlap(kk, n_shot)
                elif m_test == 'swap':
                    shot_outcomes[k] = sk_swap(kk, n_shot)

                k += 1
                
                # kk_est = np.mean(sk_overlap(kk, n_shot))
                # kernel_est[i][j] = kk_est
                # kernel_est[j][i] = kk_est
                
    return kernel_val, shot_outcomes #kernel_est


def kernel_matrix_est_only(X, kernel_, n_shot, m_test):
    N = len(X)
    #kernel_est = np.zeros((N,N))
    shot_outcomes = np.zeros((int(N*(N-1)/2),n_shot))
    k = 0
    for i in range(N):
        for j in range(i,N):
            if i != j:
                kk = kernel_(X[i],X[j])
                
                if m_test == 'overlap':
                    shot_outcomes[k] = sk_overlap(kk, n_shot)
                elif m_test == 'swap':
                    shot_outcomes[k] = sk_swap(kk, n_shot)

                k += 1
                
                # kk_est = np.mean(sk_overlap(kk, n_shot))
                # kernel_est[i][j] = kk_est
                # kernel_est[j][i] = kk_est
                
    return shot_outcomes #kernel_est


def gramm_estimates(shot_outcomes, n_shot, N):
    nn = len(n_shot)
    kernel_est = np.zeros((nn,N,N))
    k = 0
    for i in range(N):
        for j in range(i,N):
            for m in range(nn):
                if i == j:
                    kernel_est[m][i][i] = 1
                else: 
                  
                    kk_est = np.mean(shot_outcomes[k,:n_shot[m]])
                    kernel_est[m][i][j] = kk_est
                    kernel_est[m][j][i] = kk_est
                    
            if i != j:
                k += 1
                
    return kernel_est

def optimal_parameters(gram_matrix, y_label, lamb = 0):
    N = len(y_label)
    inverse_matrix = np.linalg.inv(gram_matrix - lamb*np.identity(N))
    return np.matmul(inverse_matrix, y_label)


def kernel_vec(x, xtrain):
    N = len(xtrain)
    kvec = np.zeros(N)
    for i in range(N):
        kvec[i] = fidelity_kernel(x, xtrain[i])
    return kvec

def kernel_vec_est(x, xtrain, n_shot, mtest):
    N = len(xtrain)
    kvec = np.zeros(N)
    shot_outcomes = np.zeros((N,n_shot))
    
    for i in range(N):
        kk = fidelity_kernel(x, xtrain[i])
        kvec[i] = kk
        if mtest == 'overlap':
            shot_outcomes[i] = sk_overlap(kk, n_shot)
        elif mtest == 'swap':
            shot_outcomes[i] = sk_swap(kk, n_shot)

        
    return kvec, shot_outcomes

def model_predict(kvec, opt_params):
    return np.dot(kvec, opt_params)
    
def mse(y1,y2):
    return np.mean((y1-y2)**2)

############################################################################################################
###########################
########################### Let's do some job
###########################
############################################################################################################

seed1 = 14
seed2 = 22

np.random.seed(seed1)

qubit_tot = [5,7,10,15,20,30,40]
nq = len(qubit_tot)
sample = 25

num_pair = int(sample*(sample-1)/2)
kernel_tot = np.zeros((len(qubit_tot),num_pair))


for i in range(len(qubit_tot)):
    print('---- qubits = %i----'%qubit_tot[i])
    
    xtrain, xtest = get_random_dataset(qubit_tot[i], sample, 20)
    
    init_kernel = lambda x1, x2: fidelity_kernel(x1,x2)
    kernel_tot[i] = kernel_trains(xtrain, init_kernel)

varr_ker = np.var(kernel_tot,axis=1)
mean_ker = np.mean(kernel_tot,axis=1)
plt.semilogy(qubit_tot,mean_ker, '-rs')
plt.show()
plt.semilogy(qubit_tot,varr_ker, '-o')
plt.show()
#plt.semilogy(np.arange(len(kk)),kk,'-o')
#plt.axhline(y=mkk, color='r', linestyle='-')
#plt.axhline(y=mekk, color='g', linestyle='-')

###############################################################################

np.random.seed(seed2)

num_qubit = qubit_tot
ker_qubit = mean_ker


nd = num_pair
nr = 1


num_shots = int(2e6) #40000000 #100000000 

dim_qubit = np.zeros((len(num_qubit)))
for i in range(len(num_qubit)):
    dim_qubit[i] = 2**num_qubit[i]

kernel_exact = np.zeros((len(num_qubit)))


##################################################################################

# for i in range(len(num_qubit)):
#     ss = sk_swap(ker_qubit[i], num_shots)
#     store_datum[i] = ss

outcomes_pval = np.zeros((len(num_qubit), nd, nr, num_shots))
dim_qubit = np.zeros((len(num_qubit)))


for i in range(len(num_qubit)):
    dim_qubit[i] = 2**num_qubit[i]


##################################################################################

for i in range(len(num_qubit)):
    for j in range(nd):
        for k in range(nr):
            ss = sk_swap_pval(kernel_tot[i,j], num_shots)
            outcomes_pval[i][j][k] = ss



#shot_budgets = [50,100,500,1000,5000,10000,50000,100000,500000,1000000, 5000000, 10000000, 50000000,100000000]
#shot_budgets = [5,10,50,100,500,1000,5000,10000,50000,100000,500000,1000000, 5000000, 10000000, 20000000, 40000000]
#shot_budgets = [5,10,50,100,500,1000,5000,10000,50000,100000,500000,1000000, 5000000, 10000000, 50000000, int(7e7), int(1e8), int(3e8)]

def get_shot_list(shots, div_shots=4):
    shots = int(shots)
    #shots = int(0.5e8)
    power_shots = 8 #int(np.log10(shots))
    div_shots = 4
    
    sb = [10]
    
    for i in range(1,power_shots):
        for k in range(div_shots):
            factor = 10/div_shots 
            shot_value = int((k+1)*factor*10**(i))
            
            if shot_value <= shots:
                sb.append(shot_value)
    
    nb = len(sb)
    return sb, nb


shot_budgets, nb = get_shot_list(num_shots)
shots = num_shots

if shots == int(2e6):
    shot_budgets.append(int(1.5e6))
    shot_budgets.append(int(2e6))

nb = len(shot_budgets)



###############################################################################
####################
#################### Here we also do the binomial test
####################
###############################################################################

######## case 1: all measurements

pvalues_flat = np.zeros((len(num_qubit), len(shot_budgets)))
outcomes_flat = np.reshape(outcomes_pval, (nq, nd*nr*shots))
s_counts_flat = np.zeros((nq,nb))

for i in range(nq):
    for j in range(nb):
        idx_stop = shot_budgets[j]*nr*nd
        
        s_counts_flat[i][j] = np.count_nonzero(outcomes_flat[i][:idx_stop])
        
        pvalues_flat[i][j] = stats.binom_test(s_counts_flat[i][j], n= nd*nr*shot_budgets[j], p=0.5, alternative='greater')


sbd = np.array(shot_budgets)*nd

c_pattern = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

########### plot something
for i in range(nq):
    plt.loglog(sbd, pvalues_flat[i] + 1e-10, '-o',label=num_qubit[i])
    plt.legend()
    if dim_qubit[i] < shot_budgets[-1]:
        plt.axvline(x=dim_qubit[i], linestyle='--', c = c_pattern[i])
    
    # if dim_qubit[i] < shot_budgets[-1]:
    #     plt.axvline(x=dim_qubit[i], linestyle='--', c = c_pattern[i])

plt.axhline(y=0.001, linestyle='--')
plt.show()


######## case 2: measure individuals

pval_tot = np.zeros((nq,nd,nb))

for i in range(nq):
    for j in range(nd):
        for k in range(nb):
            sc = np.count_nonzero(outcomes_pval[i][j][0][:shot_budgets[k]])
            pval_tot[i][j][k] = stats.binom_test(sc, n= shot_budgets[k], p=0.5, alternative='greater')



avg_pval = np.mean(pval_tot, axis=1)
med_pval = np.median(pval_tot, axis=1)
std_pval =  np.std(pval_tot, axis=1)

### count the success ratio
pbar = 1e-2

count_p_suc = np.zeros((nq,nb))
ratio_p_suc = np.zeros((nq,nb))

for i in range(nq):
    for j in range(nb):
        count_p_suc[i][j] = (pval_tot[i,:,j]<pbar).sum()
        ratio_p_suc[i][j] = count_p_suc[i][j]/nd



for i in range(nq):
    plt.semilogx(shot_budgets, ratio_p_suc[i], '-o', c=c_pattern[i],label=num_qubit[i])
    
    if shot_budgets[0] < dim_qubit[i] < shot_budgets[-1]:
        plt.axvline(x=dim_qubit[i], linestyle='--', c = c_pattern[i])
        
plt.xlabel('shots')
plt.ylabel('success ratio')
plt.legend()
plt.show()

np.save('new_sup_fig_pval_fidel.npy',pval_tot)
np.save('new_sup_fig_qubits.npy',num_qubit)
np.save('new_sup_fig_shots.npy',shot_budgets)
np.save('new_sup_fig_ratio_suc.npy',ratio_p_suc)


        
