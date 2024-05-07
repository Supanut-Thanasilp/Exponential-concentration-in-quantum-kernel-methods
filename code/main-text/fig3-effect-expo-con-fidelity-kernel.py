"""

Figure 3: Practical implication of exponential concentration on training and generalization (Fidelity kernel)


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


def kernel_matrix_est(X, kernel_, n_shot, m_test):
    N = len(X)
    kernel_val = np.zeros((N,N))
    shot_outcomes = np.zeros((int(N*(N-1)/2),n_shot))
    k = 0
    for i in range(N):
        for j in range(i,N):
            if i == j:
                kernel_val[i][i] = 1
            else: 
                kk = kernel_(X[i],X[j])
                kernel_val[i][j] = kk
                kernel_val[j][i] = kk
                
                if m_test == 'overlap':
                    shot_outcomes[k] = sk_overlap(kk, n_shot)
                elif m_test == 'swap':
                    shot_outcomes[k] = sk_swap(kk, n_shot)

                k += 1
                
    return kernel_val, shot_outcomes #kernel_est


def kernel_matrix_est_only(X, kernel_, n_shot, m_test):
    N = len(X)
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
                
    return shot_outcomes 


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
########################### Let's do some work
###########################
############################################################################################################

seed = 31
np.random.seed(seed)

n = 40
shots = 1000
ntest = 20
nn = np.arange(10,150,5)


measurement_test = 'swap'
measurement_test2 = 'overlap'

num_cond = np.zeros(len(nn))
norm = np.zeros(len(nn))
loss_train =  np.zeros(len(nn))
loss_test =  np.zeros(len(nn))


loss_train_est =  np.zeros(len(nn))
loss_test_est =  np.zeros(len(nn))

loss_train_est2 =  np.zeros(len(nn))
loss_test_est2 =  np.zeros(len(nn))


loss_train_rand =  np.zeros(len(nn))
loss_test_rand =  np.zeros(len(nn))



mp_test_tot = np.zeros((len(nn), ntest))
mp_test_tot_est = np.zeros((len(nn), ntest))
mp_test_tot_est2 = np.zeros((len(nn), ntest))
mp_test_tot_rand = np.zeros((len(nn), ntest))



xtrain, xtest = get_random_dataset(n, nn[-1], ntest)


mp = np.zeros((len(xtest)))
mptrain = np.zeros(len(xtrain))

mp_est = np.zeros((len(xtest)))
mptrain_est = np.zeros(len(xtrain))

mp_est2 = np.zeros((len(xtest)))
mptrain_est2 = np.zeros(len(xtrain))

mp_rand = np.zeros((len(xtest)))
mptrain_rand = np.zeros((len(xtrain)))


#np.random.seed(12) 


init_kernel = lambda x1, x2: fidelity_kernel(x1,x2)
#gramm = kernel_matrix(xtrain, init_kernel)
gramm, outcome_est = kernel_matrix_est(xtrain, init_kernel, shots, measurement_test)
gramm_est = gramm_estimates(outcome_est,[shots],nn[-1])[0]

outcome_est2 = kernel_matrix_est_only(xtrain, init_kernel, shots, measurement_test2)
gramm_est2 = gramm_estimates(outcome_est2,[shots],nn[-1])[0]



gramm_rand = random_matrix_swap(xtrain, shots)

###############################################################################
#### Attempt to get a better task
#####################################

rand_cont = 1 # 2e4
a_rand = np.random.uniform(0, rand_cont, nn[-1])

def label_generator_fidel(x_input, x_train, alpha_rand):
    
    yout = np.zeros(len(x_input))
    
    for k in range(len(x_input)):
        kvv = np.zeros(len(x_train))
        for i in range(len(x_train)):
            kvv[i] = fidelity_kernel(x_input[k], x_train[i])
        yout[k] = alpha_rand.dot(kvv)
    
    return yout

ytrain = label_generator_fidel(xtrain,xtrain, a_rand)
ytest = label_generator_fidel(xtest,xtrain,a_rand)



###############################################################################

print("phase 1 complete")

########################### Prediction phase

for k in range(len(nn)):
    A= gramm[:nn[k],:nn[k]]
    A_est = gramm_est[:nn[k],:nn[k]]
    A_est2 = gramm_est2[:nn[k],:nn[k]]
    A_rand = gramm_rand[:nn[k],:nn[k]]
    
    xx = xtrain[:nn[k]]
    yy = ytrain[:nn[k]]
    
    
    ########## exact
    
    a_opt =  optimal_parameters(A, yy)
    
    mptrain = A.dot(a_opt)
    
    for i in range(ntest):
        kv = kernel_vec(xtest[i],xx)  
        mp[i] = model_predict(kv, a_opt)
    
    loss_test[k] = mse(mp, ytest)
    loss_train[k] = mse(mptrain, yy)


    num_cond[k] = np.linalg.cond(A)
    
    A_ident = np.matmul(np.linalg.inv(A), A)
    norm[k] = np.linalg.norm(A_ident - np.identity(nn[k]))
    
    mp_test_tot[k] = mp

    ###################
    ##### est.
    
    a_est = optimal_parameters(A_est, yy)
    
    mptrain_est = A_est.dot(a_est)
    
    for i in range(ntest):
       
        kv_est =  np.mean(kernel_vec_est(xtest[i],xx, shots, measurement_test)[1],axis=1)
        mp_est[i] = model_predict(kv_est, a_est)
    
    
    mp_test_tot_est[k] = mp_est
    
    loss_test_est[k] = mse(mp_est, ytest)
    loss_train_est[k] = mse(mptrain_est, yy)
    
    ###################
    ##### est2.
    
    a_est2 = optimal_parameters(A_est2, yy)
    
    mptrain_est2 = A_est2.dot(a_est2)
    
    for i in range(ntest):
       
        kv_est2 =  np.mean(kernel_vec_est(xtest[i],xx, shots, measurement_test2)[1],axis=1)
        mp_est2[i] = model_predict(kv_est2, a_est2)
    
    
    mp_test_tot_est2[k] = mp_est2
    
    loss_test_est2[k] = mse(mp_est2, ytest)
    loss_train_est2[k] = mse(mptrain_est2, yy)

    
    ###################
    ##### rand.
    
    a_rand = optimal_parameters(A_rand, yy)
    
    mptrain_rand = A_rand.dot(a_rand)
    
    for i in range(ntest):
       
        kv_rand =  random_vec_swap(xx, shots)
        mp_rand[i] = model_predict(kv_rand, a_rand)
    
    mp_test_tot_rand[k] = mp_rand
    
    loss_test_rand[k] = mse(mp_rand,ytest)
    loss_train_rand[k] = mse(mptrain_rand,yy)

    
    

    
########################### Visualizing the training results 


plt.plot(nn[:-1],loss_train[:-1],"-bs",mec='black',label='exact', alpha=0.7)
plt.plot(nn[:-1],loss_train_est[:-1],"-rs",mec='black',label='est. (SWAP)', alpha=0.7)
plt.plot(nn[:-1],loss_train_est2[:-1],"-ro",mec='black',label='est. (overlap)', alpha=0.7)
plt.plot(nn[:-1],loss_train_rand[:-1],"-go",mec='black',label='random', alpha=0.7)
#plt.axvline(x=int(2**(n/2)), linestyle=':')
plt.legend()
plt.xlabel('training data')
plt.ylabel('train loss')
plt.ylim(-1e-3, 1e-3)
plt.show()


plt.semilogy(nn[:-1],loss_test[:-1]/loss_test[0],"-bs",mec='black',label='exact')
plt.semilogy(nn[:-1],loss_test_est[:-1]/loss_test_est[0],"-rs",mec='black',label='est. (SWAP)')
plt.semilogy(nn[:-1],loss_test_est2[:-1]/loss_test_est2[0],"-ro",mec='black',label='est. (overlap)')
plt.semilogy(nn[:-1],loss_test_rand[:-1]/loss_test_rand[0],"-go",mec='black',label='random')
#plt.axvline(x=int(2**(n/2)), linestyle=':')
plt.legend()
plt.xlabel('training data')
plt.ylabel('relative loss (on test data)')
#plt.ylim(1e-6,2e2)
plt.show()



np.save('new_main_fig_gen_loss_test_sd%i'%seed,loss_test)
np.save('new_main_fig_gen_loss_test_est_swap_sd%i'%seed,loss_test_est)
np.save('new_main_fig_gen_loss_test_est_overlap_sd%i'%seed,loss_test_est2)
np.save('new_main_fig_gen_loss_test_rand_sd%i'%seed,loss_test_rand)

np.save('new_main_fig_gen_loss_train_sd%i'%seed,loss_test)
np.save('new_main_fig_gen_loss_train_est_swap_sd%i'%seed,loss_train_est)
np.save('new_main_fig_gen_loss_train_est_overlap_sd%i'%seed,loss_train_est2)
np.save('new_main_fig_gen_loss_train_rand_sd%i'%seed,loss_train_rand)

































        
        
        
        
        
        
        
        
