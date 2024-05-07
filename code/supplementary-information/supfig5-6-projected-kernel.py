"""

Projected kernel (SWAP and local tomography)


Supplementary Figure 5: Effect of exponential concentration on estimated Gram matrix 

and 

Supplementary Figure 6: Effect of exponential concentration on training and generalization

"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from numpy import linalg as LA


#### obtain datasets

sigma_x = np.array([[0,1],[1,0]],dtype=np.complex128)
sigma_y = np.array([[0,-1j],[1j,0]],dtype=np.complex128)
sigma_z = np.array([[1,0],[0,-1]],dtype=np.complex128)
sigma_0 = np.array([[1,0],[0,1]],dtype=np.complex128)

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

def f_norm(A,B):
    diff = A - B
    return np.square(LA.norm(diff, ord='fro'))


def reduce_state(x):
    cx = np.cos(x/2)
    sx = np.sin(x/2)
    return np.array([[cx**2, cx*sx],[cx*sx, sx**2]])


def norm2_mix(rho):
    mix = np.array([[1/2,0],[0,1/2]])
    return f_norm(rho, mix)

def purity(rho):
    return np.trace(rho.dot(rho))

def overlap(rho1, rho2):
    return np.trace(rho1.dot(rho2))


def set_rho(xvec):
    n = len(xvec)
    rhovec = np.zeros((n,2,2))
    for i in range(n):
        rhovec[i] = reduce_state(xvec[i])
    return rhovec


############################################################################################################
###########################
########################### Let's first benchmark the code
###########################
############################################################################################################

from scipy.stats import unitary_group

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




def pauli_coeff(rdm1):
    cx = np.trace(rdm1.dot(sigma_x))
    cy = np.trace(rdm1.dot(sigma_y))
    cz = np.trace(rdm1.dot(sigma_z))
    return np.real(cx), np.real(cy), np.real(cz)

def reconstruct_rdm(coeff):
    cx, cy, cz = coeff
    return 0.5*sigma_0 + 0.5*cx*sigma_x + 0.5*cy*sigma_y + 0.5*cz*sigma_z

def get_coeff_from_rdm(rdm, num_qubit):
    "rdm must respect structure len(nun_qubit), data, n, 2, 2  --> rdm & num_qubit must be lists "
    nq = len(num_qubit)
    
    coeff_tot = []
    ntrain = len(rdm[0])
    
    for k in range(nq):
        #print(num_qubit[k])
        
        rho_reduce = rdm[k]
        
        coeff = np.zeros((ntrain, num_qubit[k], 3))
        
        for i in range(ntrain):
            for l in range(num_qubit[k]):
                coeff[i][l] = pauli_coeff(rho_reduce[i][l])
    
        
        coeff_tot.append(coeff)
        
    return coeff_tot
        
def get_rdm_from_coeff(coeff, num_qubit):
    "coeff must respect structure len(nun_qubit), data, n, 3 --> coeff & num_qubit must be lists "
    
    nq = len(num_qubit)
    
    rho_tot = []
    ntrain = len(coeff[0])
    
    for k in range(nq):
        
        coeff_pauli = coeff[k]
        
        rho_reduce = np.zeros((ntrain,num_qubit[k],2,2),dtype=np.complex128)
        
        for i in range(ntrain):
            for l in range(num_qubit[k]):
                rho_reduce[i][l] = reconstruct_rdm(coeff_pauli[i][l])
        
        rho_tot.append(rho_reduce)
    
    return rho_tot


def sk_swap_pval(kerr, n_sample):
    prob_p = 1/2 + kerr/2
    prob_m = 1/2 - kerr/2
    samples = np.random.choice([0,1], n_sample, p=[prob_m,prob_p])
    return samples


def get_puol_from_rdm(rdm, num_qubit):
    "rdm must respect structure len(nun_qubit), data, n, 2, 2  --> rdm & num_qubit must be lists "
    nq = len(num_qubit)
    
    puol_tot = []
    ntrain = len(rdm[0])
    
    for k in range(nq):
        print(num_qubit[k])
        
        rho_reduce = rdm[k]
        
        puol = np.zeros((ntrain, ntrain, num_qubit[k]))
        
        for i in range(ntrain):
            for j in range(i,ntrain):
                if i == j:
                    
                    for l in range(num_qubit[k]):
                        puol[i][i][l] = purity(rho_reduce[i][l])
                
                else:
                    
                    for l in range(num_qubit[k]):
                        rho1 = rho_reduce[i][l]
                        rho2 = rho_reduce[j][l]
                        
                        ovl = overlap(rho1, rho2)
                        
                        puol[i][j][l] = ovl
                        puol[j][i][l] = ovl

        puol_tot.append(puol)
        
    return puol_tot


############################################################################################################
###########################
########################### Let's do some work
###########################
############################################################################################################

### color for some plots later
c_pattern = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']


############################################################################################################
###########################
########################### Supplementary Figure 5: P-Values for SWAP and local tomography
###########################
############################################################################################################

supfig5 = False ### this is to run a code to get data for the supplementary figure 5
if supfig5:
    
    num_qubit = np.array([7,8,9,10,11,12]) #,10]) #,11,12])
    dim = [2**7,2**8,2**9 ,2**10,2**11,2**12]
    
    nq  = len(num_qubit)
    ntrain = 25
    
    rdm_train_tot = []
    
    
    np.random.seed(234)
    
    for k in range(nq):
        psi_tot = unitary_group.rvs(dim[k])[:ntrain]
        rdm_train = np.zeros((ntrain, num_qubit[k], 2, 2), dtype = np.complex128)
    
        for i in range(ntrain):
            rho = np.outer(psi_tot[i], np.conjugate(psi_tot[i])) #### potential bottleneck
            
            for l in range(num_qubit[k]):
                rdm_train[i][l] = partial_trace(rho, [l] )
    
        rdm_train_tot.append(rdm_train)
    
    
    puol_tot = get_puol_from_rdm(rdm_train_tot, num_qubit)
    coeff_tot = get_coeff_from_rdm(rdm_train_tot, num_qubit)
    
    
    ###############################################################################
    
    ### Initiate variables
    outcomes_pval_puol_tot = []
    outcomes_pval_coeff_tot = []
    
    
    ### Sample something
    
    shots = int(1e5)
    
    for k in range(nq):
        print("sample process at qubit = ", num_qubit[k])
        n = num_qubit[k]
        
        outcomes_pval_puol = np.zeros((ntrain, ntrain, num_qubit[k],shots))
        outcomes_pval_coeff = np.zeros((ntrain,num_qubit[k],3,shots))
        
        puol = puol_tot[k]
        coeff = coeff_tot[k]
        
        for i in range(ntrain):
            for j in range(i,ntrain):
                for l in range(num_qubit[k]):
                    
                    ss = sk_swap_pval(puol[i][j][l], shots)
                    
                    if i == j:
                        outcomes_pval_puol[i][j][l] = ss
                    else:
                        outcomes_pval_puol[i][j][l] = ss
                        outcomes_pval_puol[j][i][l] = ss
            
            for ll in range(num_qubit[k]):
                for pp in range(3):
                    ssr = sk_swap_pval(coeff[i][ll][pp], shots)
                    outcomes_pval_coeff[i][ll][pp] = ssr
                    
        
        outcomes_pval_puol_tot.append(outcomes_pval_puol)
        outcomes_pval_coeff_tot.append(outcomes_pval_coeff)
    
    ### save some data
    # for k in range(nq):
    #     np.save('new_sup_fig_pqk_swap_qubit%i'%num_qubit[k],outcomes_pval_puol_tot[k])
    
    ###############################################################################
    
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
    
    
    
    shot_budgets, nb = get_shot_list(shots)
    
    
    ###############################################################################
    
    pval_puol_tot = []
    pval_coeff_tot = []
    
    for k in range(nq):
        print("binomial test process at qubit = ", num_qubit[k])
        
        pval_puol = np.zeros((ntrain, ntrain, num_qubit[k],nb))
        pval_coeff = np.zeros((ntrain, num_qubit[k], 3, nb))
        
        for i in range(ntrain):
            for j in range(i,ntrain):
                for l in range(num_qubit[k]):
                    for jj in range(nb):
                        sc = np.count_nonzero(outcomes_pval_puol_tot[k][i][j][l][:shot_budgets[jj]])
                        if i == j:
                            pval = stats.binom_test(sc, n= shot_budgets[jj], p=0.75, alternative='greater')
                            pval_puol[i][j][l][jj] = pval
                        else:
                            pval = stats.binom_test(sc, n= shot_budgets[jj], p=0.75)
                            pval_puol[i][j][l][jj] = pval
            
            for l in range(num_qubit[k]):
                for j in range(3):
                    for jj in range(nb):
                        sc2 = np.count_nonzero(outcomes_pval_coeff_tot[k][i][l][j][:shot_budgets[jj]])
                        
                        pval2 = stats.binom_test(sc2, n= shot_budgets[jj], p=0.5)
                        pval_coeff[i][l][j][jj] = pval2
        
        pval_puol_tot.append(pval_puol)
        pval_coeff_tot.append(pval_coeff)
    
    
    
    ###############################################################################
    
    
    ## transform to a proper array
    npp = int(ntrain*(ntrain-1)/2 + ntrain)
    pval_puol_tot2 = []
    
    for k in range(nq):
        pval_puol2 = np.zeros((npp, num_qubit[k],nb))
        kk = 0
        
        for i in range(ntrain):
            for j in range(i,ntrain):
                pval_puol2[kk] = pval_puol_tot[k][i][j] 
                kk += 1
        
        pval_puol_tot2.append(pval_puol2)
        
                
    for i in range(nq):
        #np.save('pval_puol_qubit%i'%num_qubit[i], pval_puol_tot2[i])
        np.save('pval_coeff_qubit%i'%num_qubit[i], pval_coeff_tot[i])
    
    
    
    ###############################################################################
    ### count the success ratio
    pbar = 1e-2
    
    count_p_suc = np.zeros((nq,nb))
    ratio_p_suc = np.zeros((nq,nb))
    
    count_p_suc2 = np.zeros((nq,nb))
    ratio_p_suc2 = np.zeros((nq,nb))
    
    
    for i in range(nq):
        for j in range(nb):
            nd = npp*num_qubit[i]
            count_p_suc[i][j] = (pval_puol_tot2[i][:,:,j] < pbar).sum()
            ratio_p_suc[i][j] = count_p_suc[i][j]/(nd)
            
            
            nd2 = ntrain*num_qubit[i]*3
            
            count_p_suc2[i][j] = (pval_coeff_tot[i][:,:,:,j]<pbar).sum()
            ratio_p_suc2[i][j] = count_p_suc2[i][j]/nd2
    
    
    
    for i in range(nq):
        plt.semilogx(shot_budgets, ratio_p_suc[i], '-o', c = c_pattern[i],label=num_qubit[i])
        plt.axvline(dim[i],linestyle='--',c=c_pattern[i])
    plt.xlabel('shots')
    plt.ylabel('success ratio')
    
    
    np.save('ratio_suc_swap.npy', ratio_p_suc)
    np.save('ratio_suc_tomo.npy', ratio_p_suc2)


##############################################################################
##############################################################################
##############################################################################
##############################################################################



supfig6 = False ### this is to run a code to get data for the supplementary figure 6
if supfig6:
    
    n = 6 #10
    dim = 2**n
    
    nn = np.arange(2,10,1)#(10,200,5)
    ntrain = nn[-1]
    ntest = 5 #20
    
    #np.random.seed(234)
    seed = 14 #58
    np.random.seed(seed)
    
    
    psi_tot = unitary_group.rvs(dim)[:(nn[-1] + ntest)]
    psi_train_tot, psi_test_tot = psi_tot[:nn[-1]], psi_tot[nn[-1]:nn[-1] + ntest]
    
    
    ####################################
    ####################################
    
    rdm_train = np.zeros((nn[-1], n, 2, 2), dtype = np.complex128)
    rdm_test = np.zeros((ntest, n, 2, 2), dtype = np.complex128)
    
    
    for i in range(nn[-1]):
        rho = np.outer(psi_train_tot[i], np.conjugate(psi_train_tot[i])) #### potential bottleneck
        
        for l in range(n):
            rdm_train[i][l] = partial_trace(rho, [l] )
    
    
    for i in range(ntest):
        rho = np.outer(psi_test_tot[i], np.conjugate(psi_test_tot[i])) #### potential bottleneck
        
        for l in range(n):
            rdm_test[i][l] = partial_trace(rho, [l] )
    
    
    ####################################
    ###############
    ############### Construct kernel values, Gram matrix and kernel vectors
    ###############
    ####################################
    
    gamma = 1
    num_pair = int(nn[-1]*(nn[-1]-1)/2)
    
    def projected_kernel(rho1, rho2, gamma = gamma):
        diff = rho1 - rho2
        component = np.sum(np.square(LA.norm(diff, axis = (1,2), ord='fro')))
        return np.exp(-gamma*component)
        
    ker_tot = np.zeros((num_pair))
    gram_tot = np.zeros((nn[-1],nn[-1]))
    kv_tot = np.zeros((ntest, nn[-1]))
    
    kk = 0
    for i in range(nn[-1]):
        for j in range(i,nn[-1]):
            if i == j:
                gram_tot[i][i] = 1
            else:
                pqk = projected_kernel(rdm_train[i], rdm_train[j])
                gram_tot[i][j] = pqk
                gram_tot[j][i] = pqk
                ker_tot[kk] = pqk
                
                kk += 1
        
        for l in range(ntest):
            kv_tot[l][i] = projected_kernel(rdm_test[l], rdm_train[i])
    
    
    ####################################
    ###############
    ############### Find purities and overlaps
    ###############
    ####################################
    
    
    ############################################################################################################
    ############################################################################################################
    ############################################################################################################
                    
                
    coeff_train = get_coeff_from_rdm([rdm_train],[n])[0]
    coeff_test = get_coeff_from_rdm([rdm_test],[n])[0]
                
    
    ############################################################################################################
    ############################################################################################################
    ############################################################################################################
    
    
    
    
    def get_puol_from_rdm(rdm, num_qubit):
        "rdm must respect structure len(nun_qubit), data, n, 2, 2  --> rdm & num_qubit must be lists "
        nq = len(num_qubit)
        
        puol_tot = []
        ntrain = len(rdm[0])
        
        for k in range(nq):
            print(num_qubit[k])
            
            rho_reduce = rdm[k]
            
            puol = np.zeros((ntrain, ntrain, num_qubit[k]))
            
            for i in range(ntrain):
                for j in range(i,ntrain):
                    if i == j:
                        
                        for l in range(num_qubit[k]):
                            puol[i][i][l] = purity(rho_reduce[i][l])
                    
                    else:
                        
                        for l in range(num_qubit[k]):
                            rho1 = rho_reduce[i][l]
                            rho2 = rho_reduce[j][l]
                            
                            ovl = overlap(rho1, rho2)
                            
                            puol[i][j][l] = ovl
                            puol[j][i][l] = ovl
    
            puol_tot.append(puol)
            
        return puol_tot
    
    
    def get_puol_test_train(rdm1, rdm2, num_qubit):
        "rdm1 must respect structure len(num_qubit), data_test, n, 2, 2  --> rdm & num_qubit must be lists "
        "rdm2 must respect structure len(num_qubit), data_train, n, 2, 2  --> rdm & num_qubit must be lists "
    
        nq = len(num_qubit)
        puol_vec_tot = []
        
        ntest = len(rdm1[0])
        ntrain = len(rdm2[0])
    
        
        for k in range(nq):
            rho_reduce1 = rdm1[k]
            rho_reduce2 = rdm2[k]
            
            puol_vec = np.zeros((ntest,ntrain,num_qubit[k],3))
            
            for i in range(ntest):
                for j in range(ntrain):
                    for l in range(num_qubit[k]):
                        pur1 = purity(rho_reduce1[i][l])
                        pur2 = purity(rho_reduce2[j][l])
                        ovel = overlap(rho_reduce1[i][l], rho_reduce2[j][l])
                        
                        puol_vec[i,j,l,0] = pur1
                        puol_vec[i,j,l,1] = pur2
                        puol_vec[i,j,l,2] = ovel
            
            puol_vec_tot.append(puol_vec)
            
        return puol_vec_tot
    
    
    
    puol_train = get_puol_from_rdm([rdm_train], [n])[0]
    puol_test = get_puol_test_train([rdm_test],[rdm_train], [n])[0]
    
    
    ## Sanity check
    # kerr = np.zeros((ntrain, ntrain))
    # for i in range(nn[-1]):
    #     for j in range(i,nn[-1]):
    #         if i == j:
    #             kerr[i][i] = 0
    #         else:
                
    #             pqk = np.sum(puol_train[i][i] + puol_train[j][j] - 2 *puol_train[i][j])
                
    #             kerr[i][j] = pqk
    #             kerr[j][i] = pqk
                
    # pqkk = np.exp(-gamma*kerr)
    #
    # kerr = np.zeros((ntest, ntrain))
    
    # for i in range(ntest):
    #     for j in range(ntrain):
    #         #pqk = 0
    #         # for l in range(n):
    #         #     pqk += puol_test[i][j][l][0] + puol_test[i][j][l][1] - 2*puol_test[i][j][l][2]
                
    #         kerr[i][j] = np.sum(puol_test[i][j][:,0] + puol_test[i][j][:,1] - 2*puol_test[i][j][:,2])
            
    # pqkk =  np.exp(-gamma*kerr)
    ### Agreed with the other approach
               
                
    ###############################################################################
    ###############                         #######################################
    ############### Let's do some samples   #######################################
    ###############                         #######################################
    ###############################################################################
    
    shots = int(1e3)#int(1e3)
    
    ####################################
    ###############
    ############### SWAP test
    ###############
    ####################################
    
    outcome_puol_train = np.zeros((ntrain,ntrain,n,shots))
    outcome_puol_test = np.zeros((ntest, ntrain, n, 3, shots))
    
    outcome_puol_train_rand = np.zeros((ntrain,ntrain,n,shots))
    outcome_puol_test_rand = np.zeros((ntest, ntrain, n, 3, shots))
    
    
    for i in range(ntrain):
        for j in range(i,ntrain):
            for k in range(n):
                outcome_puol_train[i][j][k] = sk_swap(puol_train[i][j][k],shots)
                outcome_puol_train_rand[i][j][k] = sk_swap(1/4,shots)
        
        for l in range(ntest):
            for k in range(n):
                outcome_puol_test[l][i][k][0] = sk_swap(puol_test[l][i][k][0],shots)
                outcome_puol_test[l][i][k][1] = sk_swap(puol_test[l][i][k][1],shots)
                outcome_puol_test[l][i][k][2] = sk_swap(puol_test[l][i][k][2],shots)
                
                outcome_puol_test_rand[l][i][k][0] = sk_swap(1/2,shots)
                outcome_puol_test_rand[l][i][k][1] = sk_swap(1/2,shots)
                outcome_puol_test_rand[l][i][k][2] = sk_swap(1/2,shots)
                
    
    
    ### Here the number of shots is fixed
    
    
    puol_train_est = np.mean(outcome_puol_train, axis = 3)
    puol_test_est = np.mean(outcome_puol_test, axis = 4)
    
    puol_train_rand = np.mean(outcome_puol_train_rand, axis = 3)
    puol_test_rand = np.mean(outcome_puol_test_rand, axis = 4)
    
    
    c_train_est = np.zeros((ntrain, ntrain))
    c_train_rand = np.zeros((ntrain, ntrain))
    
    for i in range(nn[-1]):
        for j in range(i,nn[-1]):
            if i == j:
                c_train_est[i][i] = 0
                c_train_rand[i][i] = 0
            else:
                
                pqk = np.sum(puol_train_est[i][i] + puol_train_est[j][j] - 2 *puol_train_est[i][j])
                
                c_train_est[i][j] = pqk
                c_train_est[j][i] = pqk
                
                pqk_rand = np.sum(puol_train_rand[i][i] + puol_train_rand[j][j] - 2 *puol_train_rand[i][j])
                
                c_train_rand[i][j] = pqk_rand
                c_train_rand[j][i] = pqk_rand
                
    
    gram_est_tot = np.exp(-gamma*c_train_est)
    gram_rand_tot = np.exp(-gamma*c_train_rand)
    
    
    
    c_test_est = np.zeros((ntest, ntrain))
    c_test_rand = np.zeros((ntest, ntrain))
    
    for i in range(ntest):
        for j in range(ntrain):
                
            c_test_est[i][j] = np.sum(puol_test_est[i][j][:,0] + puol_test_est[i][j][:,1] - 2*puol_test_est[i][j][:,2])
            c_test_rand[i][j] = np.sum(puol_test_rand[i][j][:,0] + puol_test_rand[i][j][:,1] - 2*puol_test_rand[i][j][:,2])
    
            
    kv_est_tot =  np.exp(-gamma*c_test_est)
    kv_rand_tot =  np.exp(-gamma*c_test_rand)
    
    
    ####################################
    ###############
    ############### Full tomography
    ###############
    ####################################
    
    
    
    outcome_coeff_train = np.zeros((ntrain,n,3,shots))
    outcome_coeff_test = np.zeros((ntest,n,3,shots))
    
    outcome_coeff_train_rand = np.zeros((ntrain,n,3,shots))
    outcome_coeff_test_rand = np.zeros((ntest,n,3,shots))
    
    for i in range(ntrain):
        for j in range(n):
            for k in range(3):
                outcome_coeff_train[i][j][k] = sk_swap(coeff_train[i][j][k], shots)
                outcome_coeff_train_rand[i][j][k] = sk_swap(0, shots)
          
                
          
    for i in range(ntest):
        for j in range(n):
            for k in range(3):
                outcome_coeff_test[i][j][k] = sk_swap(coeff_test[i][j][k], shots)
                outcome_coeff_test_rand[i][j][k] = sk_swap(0, shots)
           
    
    ### Here samples are fixed
    
    coeff_train_est = np.mean(outcome_coeff_train, axis=3)
    coeff_test_est = np.mean(outcome_coeff_test, axis=3)
    
    coeff_train_rand = np.mean(outcome_coeff_train_rand, axis=3)
    coeff_test_rand = np.mean(outcome_coeff_test_rand, axis=3)
    
    
    # Reconstruct the estimated rdm
    
    rdm_train_est = get_rdm_from_coeff([coeff_train_est], [n])[0]
    rdm_test_est = get_rdm_from_coeff([coeff_test_est], [n])[0]
    
    rdm_train_rand = get_rdm_from_coeff([coeff_train_rand], [n])[0]
    rdm_test_rand = get_rdm_from_coeff([coeff_test_rand], [n])[0]
    
    
    # Reconstruct the estimated gram matrix
    
    gram_est_tot2 = np.zeros((nn[-1],nn[-1]))
    kv_est_tot2 = np.zeros((ntest, nn[-1]))
    
    gram_rand_tot2 = np.zeros((nn[-1],nn[-1]))
    kv_rand_tot2 = np.zeros((ntest, nn[-1]))
    
    
    for i in range(nn[-1]):
        for j in range(i,nn[-1]):
            if i == j:
                gram_est_tot2[i][i] = 1
                gram_rand_tot2[i][i] = 1
            else:
                pqk_est = projected_kernel(rdm_train_est[i], rdm_train_est[j])
                gram_est_tot2[i][j] = pqk_est
                gram_est_tot2[j][i] = pqk_est
                
                pqk_rand2 = projected_kernel(rdm_train_rand[i], rdm_train_rand[j])
                gram_rand_tot2[i][j] = pqk_rand2
                gram_rand_tot2[j][i] = pqk_rand2
    
        
        for l in range(ntest):
            kv_est_tot2[l][i] = projected_kernel(rdm_test_est[l], rdm_train_est[i])
            kv_rand_tot2[l][i] = projected_kernel(rdm_test_rand[l], rdm_train_rand[i])
    
    
    
    ###############################################################################
    ###############                          ######################################
    ############### Construct a fakce task   ######################################
    ###############                          ######################################
    ###############################################################################
    
    
    rand_cont = 1 # 2e4
    a_rand = np.random.uniform(0, rand_cont, nn[-1])
    
    ytrain = a_rand.dot(gram_tot)
    ytest = kv_tot.dot(a_rand)
        
    
    ####################################
    ###############
    ############### Train model on exact kernel
    ###############
    ####################################
    
    print("phase 1 complete")
    
    ########################### Prediction phase
    
    mp = np.zeros(ntest)
    loss_test = np.zeros(len(nn))
    loss_train = np.zeros(len(nn))
    
    
    ### SWAP test
    
    mp_est = np.zeros(ntest)
    mp_test_est_tot = np.zeros((len(nn),ntest))
    loss_test_est = np.zeros(len(nn))
    loss_train_est = np.zeros(len(nn))
    
    mp_rand = np.zeros(ntest)
    mp_test_rand_tot = np.zeros((len(nn),ntest))
    loss_test_rand = np.zeros(len(nn))
    loss_train_rand = np.zeros(len(nn))
    
    ### Tomography test
    
    mp_est2 = np.zeros(ntest)
    mp_test_est_tot2 = np.zeros((len(nn),ntest))
    loss_test_est2 = np.zeros(len(nn))
    loss_train_est2 = np.zeros(len(nn))
    
    mp_rand2 = np.zeros(ntest)
    mp_test_rand_tot2 = np.zeros((len(nn),ntest))
    loss_test_rand2 = np.zeros(len(nn))
    loss_train_rand2 = np.zeros(len(nn))
    
    
    
    
    for k in range(len(nn)):
        A= gram_tot[:nn[k],:nn[k]]
        A_est = gram_est_tot[:nn[k],:nn[k]]
        A_est2 = gram_est_tot2[:nn[k],:nn[k]]
        
        A_rand = gram_rand_tot[:nn[k],:nn[k]]
        A_rand2 = gram_rand_tot2[:nn[k],:nn[k]]
    
        
        kv = kv_tot[:,:nn[k]]
        kv_est = kv_est_tot[:,:nn[k]]
        kv_est2 = kv_est_tot2[:,:nn[k]]
        
        kv_rand = kv_rand_tot[:,:nn[k]]
        kv_rand2 = kv_rand_tot2[:,:nn[k]]
    
    
        
        yy = ytrain[:nn[k]]
        
        
        ########## exact
        
        a_opt =  optimal_parameters(A, yy)
        
        mptrain = A.dot(a_opt)
        
        for i in range(ntest):
            mp[i] = model_predict(kv[i], a_opt)
        
        loss_test[k] = mse(mp, ytest)
        loss_train[k] = mse(mptrain, yy)
    
    
    
        # num_cond[k] = np.linalg.cond(A)
        
        # A_ident = np.matmul(np.linalg.inv(A), A)
        # norm[k] = np.linalg.norm(A_ident - np.identity(nn[k]))
        
        # mp_test_tot[k] = mp
    
        ###################
        ##### est.
        
        a_est = optimal_parameters(A_est, yy)
        
        mptrain_est = A_est.dot(a_est)
        
        for i in range(ntest):
            mp_est[i] = model_predict(kv_est[i], a_est)
        
        
        mp_test_est_tot[k] = mp_est
        
        loss_test_est[k] = mse(mp_est, ytest)
        loss_train_est[k] = mse(mptrain_est, yy)
        
        ###################
        ##### est2.
        
        a_est2 = optimal_parameters(A_est2, yy)
        
        mptrain_est2 = A_est2.dot(a_est2)
        
        for i in range(ntest):
           
            mp_est2[i] = model_predict(kv_est2[i], a_est2)
        
        
        mp_test_est_tot2[k] = mp_est2
        
        loss_test_est2[k] = mse(mp_est2, ytest)
        loss_train_est2[k] = mse(mptrain_est2, yy)
    
        
        ###################
        ##### rand. 1
        
        a_rand = optimal_parameters(A_rand, yy)
        
        mptrain_rand = A_rand.dot(a_rand)
        
        for i in range(ntest):
            mp_rand[i] = model_predict(kv_rand[i], a_rand)
        
        mp_test_rand_tot[k] = mp_rand
        
        loss_test_rand[k] = mse(mp_rand,ytest)
        loss_train_rand[k] = mse(mptrain_rand,yy)
        
        
        
        ###################
        ##### rand. 2
        
        a_rand2 = optimal_parameters(A_rand2, yy)
        
        mptrain_rand2 = A_rand2.dot(a_rand2)
        
        for i in range(ntest):
            mp_rand2[i] = model_predict(kv_rand2[i], a_rand2)
        
        mp_test_rand_tot2[k] = mp_rand2
        
        loss_test_rand2[k] = mse(mp_rand2,ytest)
        loss_train_rand2[k] = mse(mptrain_rand2,yy)
    
        
    # plt.semilogy(nn, loss_train+1e-6, '-o')
    
    plt.plot(nn[:-1],loss_train[:-1],"-bs",mec='black',label='exact', alpha=0.7)
    plt.plot(nn[:-1],loss_train_est[:-1],"-rs",mec='black',label='est. (SWAP)', alpha=0.7)
    plt.plot(nn[:-1],loss_train_est2[:-1],"-ro",mec='black',label='est. (overlap)', alpha=0.7)
    plt.plot(nn[:-1],loss_train_rand[:-1],"-go",mec='black',label='random', alpha=0.7)
    plt.plot(nn[:-1],loss_train_rand2[:-1],"-go",mec='black',label='random2', alpha=0.7)
    #plt.axvline(x=int(2**(n/2)), linestyle=':')
    plt.legend()
    plt.xlabel('training data')
    plt.ylabel('train loss')
    plt.ylim(-1e-3, 1e-3)
    plt.show()
    
    plt.semilogy(nn, (loss_test+1e-6)/loss_test[0], '-bs',mec='black',label='exact',alpha=0.8)
    plt.semilogy(nn, (loss_test_est+1e-6)/loss_test_est[0], '-rs',mec='black',label='est. (SWAP)',alpha=0.8)
    plt.semilogy(nn, (loss_test_est2+1e-6)/loss_test_est2[0], '-rP',mec='black',label='est. (Tomo)',alpha=0.8)
    plt.semilogy(nn, (loss_test_rand+1e-6)/loss_test_rand[0], '-go',mec='black',label='random',alpha=0.8)
    plt.semilogy(nn, (loss_test_rand2+1e-6)/loss_test_rand2[0], '-gp',mec='black',label='random2',alpha=0.8)
    #plt.semilogy(nn, loss_train_est+1e-6, '-go')
    plt.legend()
    plt.xlabel('training data')
    plt.ylabel('relative loss')
    plt.show()
    
    
    
    
    np.save('sup_fig_gen_pqk_loss_test_sd%i'%seed,loss_test)
    np.save('sup_fig_gen_pqk_loss_test_est_swap_sd%i'%seed,loss_test_est)
    np.save('sup_fig_gen_pqk_loss_test_est_tomo_sd%i'%seed,loss_test_est2)
    np.save('sup_fig_gen_pqk_loss_test_rand_swap_sd%i'%seed,loss_test_rand)
    np.save('sup_fig_gen_pqk_loss_test_rand_tomo_sd%i'%seed,loss_test_rand2)
    
    
    np.save('sup_fig_gen_pqk_loss_train_sd%i'%seed,loss_train)
    np.save('sup_fig_gen_pqk_loss_train_est_swap_sd%i'%seed,loss_train_est)
    np.save('sup_fig_gen_pqk_loss_train_est_tomo_sd%i'%seed,loss_train_est2)
    np.save('sup_fig_gen_pqk_loss_train_rand_swap_sd%i'%seed,loss_train_rand)
    np.save('sup_fig_gen_pqk_loss_train_rand_tomo_sd%i'%seed,loss_train_rand2)
    




