"""

Supplementary Figure 4a: Effect of exponential concentration on estimated Gram matrix (the Loschmidt Echo test + fidelity kernel)


"""

import numpy as np
import matplotlib.pyplot as plt



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


###############################################################################
##### Random variables
###############################################################################

def random_matrix_swap(X, n_shot):
    N = len(X)
    kernel_val = np.zeros((N,N))

    
    for i in range(N):
        for j in range(i,N):
            if i == j:
                kernel_val[i][i] = 1
            else: 
                
                kk = np.mean(sk_swap(0, n_shot))
                
                kernel_val[i][j] = kk
                kernel_val[j][i] = kk
    
    return kernel_val

def random_vec_swap(xtrain, n_shot):
    N = len(xtrain)
    kvec = np.zeros(N)
    
    for i in range(N):
        kvec[i] = np.mean(sk_swap(0, n_shot))

    return kvec

############################################################################################################
###########################
########################### Let's do some job
###########################
############################################################################################################



qubit_tot = [5,7,10,15,20,30,40]
sample = 25

num_pair = int(sample*(sample-1)/2)
kernel_tot = np.zeros((len(qubit_tot),num_pair))

for i in range(len(qubit_tot)):
    print('---- qubits = %i----'%qubit_tot[i])
    
    xtrain, xtest = get_random_dataset(qubit_tot[i], sample, 20)
    
    init_kernel = lambda x1, x2: fidelity_kernel(x1,x2)
    kernel_tot[i] = kernel_trains(xtrain, init_kernel)



##################################################################################
################################################################################## Measure something
##################################################################################

np.random.seed(58)

num_qubit = qubit_tot 

nr = 1 # num repeat
nq = len(num_qubit)
nd = num_pair

num_shots = 100000 #int(2e6) 
shots = num_shots

store_datum = np.zeros((len(num_qubit), nd, nr, num_shots))
dim_qubit = np.zeros((len(num_qubit)))


for i in range(len(num_qubit)):
    dim_qubit[i] = 2**num_qubit[i]


##################################################################################

for i in range(len(num_qubit)):
    for j in range(nd):
        for k in range(nr):
            ss = sk_overlap(kernel_tot[i,j], num_shots)
            store_datum[i][j][k] = ss



###############################################################################

def get_shot_list(shots, div_shots=4):
    shots = int(shots)
    power_shots = 8 
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

if shots == int(2e6):
    shot_budgets.append(int(1.5e6))
    shot_budgets.append(int(2e6))

nb = len(shot_budgets)

###############################################################################


kernel_estimates = np.zeros((len(num_qubit), nd, nr, len(shot_budgets)))



for k in range(len(shot_budgets)):
    ns = shot_budgets[k]

        
for k in range(len(shot_budgets)):
    ns = shot_budgets[k]
    kernel_estimates[:,:,:,k] = np.mean(store_datum[:,:,:,:ns],axis=3)

mean1_est = np.mean(kernel_estimates, axis=2)
std1_est = np.std(kernel_estimates, axis=2)


kernel_rel_tot = np.zeros((nq,nd,nb))

for i in range(nb):
    kernel_rel_tot[:,:,i] = kernel_tot


rel_error = np.abs((kernel_rel_tot - mean1_est))/kernel_rel_tot
abs_error = np.abs((kernel_rel_tot - mean1_est))

avg_rel_error = np.mean(rel_error, axis=1)
avg_abs_error = np.mean(abs_error, axis=1)

std_rel_error = np.std(rel_error,axis=1)
std_abs_error = np.std(abs_error,axis=1)



###############################################################################


ntrain = sample
count_acc = np.zeros((nq,nb))
count_zer = np.zeros((nq,nb))

ratio_acc = np.zeros((nq,nb))
ratio_zer = np.zeros((nq,nb))

pbar = 1e-1
npp = int(ntrain*(ntrain-1)/2)


for i in range(nq):
    for j in range(nb):
        count_acc[i][j] = (rel_error[i,:,j]<pbar).sum() 
        ratio_acc[i][j] = count_acc[i][j]/npp
        
        count_zer[i][j] = npp - np.count_nonzero(rel_error[i,:,j] - 1)
        ratio_zer[i][j] = count_zer[i][j]/npp


c_pattern = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

for i in range(nq):
    plt.semilogx(shot_budgets, ratio_zer[i], '-o', c = c_pattern[i])
    if shot_budgets[0] < dim_qubit[i] < shot_budgets[-1]:
        plt.axvline(x=dim_qubit[i], linestyle='--', c = c_pattern[i])
plt.xlabel('shots')
plt.ylabel('zero ratio')
plt.show()

for i in range(nq):
    plt.semilogx(shot_budgets, ratio_acc[i], '-o', c = c_pattern[i])
    if shot_budgets[0] < dim_qubit[i] < shot_budgets[-1]:
        plt.axvline(x=dim_qubit[i], linestyle='--', c = c_pattern[i])
plt.xlabel('shots')
plt.ylabel('accuracy ratio with relative error less than 0.1')
plt.show()


np.save('fig_fidel_overlap_rel_error.npy',rel_error)
np.save('fig_fidel_overlap_zero_ratio.npy',ratio_zer)
np.save('fig_fidel_overlap_acc_ratio.npy',ratio_acc)


