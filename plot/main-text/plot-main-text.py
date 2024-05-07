"""

This is to plot figures in the main text. Data have to already be stored in the folder. 


"""

# import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


plt.rc('text', usetex=True)
plt.rc('font', family='serif')

###############################################################################
###################### Figure 3 : Generalzation as a function of training data
###############################################################################

colors = ['#7F8EC9','#4056A1','#075C2F','#7D8238','#453F3F','#692411','#D79922','#F13C20','black']
markers = ['-o','-s','-^','-D','-P','-p','-*']


fig3_plot_main = True

if fig3_plot_main:


    seed = [3,6,7,10,11,14,15,23,31,450]
    
    
    nn = np.arange(10,150,5)
    
    nsd = len(seed)
    ns = len(nn)
    
    loss_test_tot = np.zeros((nsd, ns))
    loss_test_est_tot = np.zeros((nsd, ns))
    loss_test_est_tot2 = np.zeros((nsd, ns))
    loss_test_rand_tot = np.zeros((nsd, ns))
    
    loss_train_tot = np.zeros((nsd, ns))
    loss_train_est_tot = np.zeros((nsd, ns))
    loss_train_est_tot2 = np.zeros((nsd, ns))
    loss_train_rand_tot = np.zeros((nsd, ns))
    
    
    mk_dir = 'effect_expo_gen/' 
    
    for i in range(nsd):
        ls =  np.load(mk_dir+'new_main_fig_gen_loss_test_sd%i.npy'%seed[i])
        loss_test_tot[i] = ls/ls[0]
    
        ls =  np.load(mk_dir+'new_main_fig_gen_loss_test_est_swap_sd%i.npy'%seed[i])
        loss_test_est_tot[i] = ls/ls[0]
        
        ls = np.load(mk_dir+'new_main_fig_gen_loss_test_est_overlap_sd%i.npy'%seed[i])
        loss_test_est_tot2[i] = ls/ls[0]
        
        ls = np.load(mk_dir+'new_main_fig_gen_loss_test_rand_sd%i.npy'%seed[i])
        loss_test_rand_tot[i] = ls/ls[0]
    
    
        ls =  np.load(mk_dir+'new_main_fig_gen_loss_train_sd%i.npy'%seed[i])
        loss_train_tot[i] = ls
    
        ls =  np.load(mk_dir+'new_main_fig_gen_loss_train_est_swap_sd%i.npy'%seed[i])
        loss_train_est_tot[i] = ls
        
        ls = np.load(mk_dir+'new_main_fig_gen_loss_train_est_overlap_sd%i.npy'%seed[i])
        loss_train_est_tot2[i] = ls
        
        ls = np.load(mk_dir+'new_main_fig_gen_loss_train_rand_sd%i.npy'%seed[i])
        loss_train_rand_tot[i] = ls
    
    
    
    
    loss_test = np.mean(loss_test_tot, axis=0)
    loss_test_est =  np.mean(loss_test_est_tot, axis=0)
    loss_test_est2 =  np.mean(loss_test_est_tot2, axis=0)
    loss_test_rand =  np.mean(loss_test_rand_tot, axis=0)
    
    loss_train = np.mean(loss_train_tot, axis=0)
    loss_train_est =  np.mean(loss_train_est_tot, axis=0)
    loss_train_est2 =  np.mean(loss_train_est_tot2, axis=0)
    loss_train_rand =  np.mean(loss_train_rand_tot, axis=0)
    
    
    sd_loss_test = np.std(loss_test_tot, axis=0)
    sd_loss_test_est =  np.std(loss_test_est_tot, axis=0)
    sd_loss_test_est2 =  np.std(loss_test_est_tot2, axis=0)
    sd_loss_test_rand =  np.std(loss_test_rand_tot, axis=0)
    
    sd_loss_train = np.std(loss_train_tot, axis=0)
    sd_loss_train_est =  np.std(loss_train_est_tot, axis=0)
    sd_loss_train_est2 =  np.std(loss_train_est_tot2, axis=0)
    sd_loss_train_rand =  np.std(loss_train_rand_tot, axis=0)
    
    
    
    fig1 = plt.figure(constrained_layout=True) #,figsize=(4.5,2.25))
    fig1.set_constrained_layout_pads(w_pad=2./72., h_pad=2./72., hspace=0., wspace=0.)
    ax_fig1 = fig1.add_subplot()
    
    ax_fig1.plot(nn[:-1],loss_test[:-1],'-o',mec='black',label='exact', c = colors[6], alpha=0.9)
    ax_fig1.plot(nn[:-1],loss_test_est[:-1],"-D",mec='black',label='est. (SWAP)', c = colors[1], alpha=0.9)
    ax_fig1.plot(nn[:-1],loss_test_est2[:-1],"-s",mec='black',label='est. (Loschmidt Echo)', c = colors[0], alpha=0.9)
    ax_fig1.plot(nn[:-1],loss_test_rand[:-1],"-^",mec='black',label='random', c = colors[3], alpha=0.9)
    ax_fig1.set_yscale('log')
    ax_fig1.tick_params(axis='y', labelsize="large",direction="in")
    ax_fig1.tick_params(axis='x', labelsize="large",direction="in")
    ax_fig1.minorticks_off()
    ax_fig1.set_ylabel(r'relative test loss, $\eta$',fontsize="x-large")
    ax_fig1.set_xlabel(r'Number of training data, $N_s$',fontsize="x-large")
    ax_fig1.legend(numpoints=1,ncol=2,loc='lower left',framealpha=1,edgecolor="black",fancybox=False, fontsize="medium", title_fontsize='large')
    
    
    
    
    ax_fig1.fill_between(nn[:-1], loss_test[:-1] - sd_loss_test[:-1], loss_test[:-1] + sd_loss_test[:-1],  color = colors[6],alpha=0.2)
    ax_fig1.fill_between(nn[:-1], loss_test_est[:-1] - sd_loss_test_est[:-1], loss_test_est[:-1] + sd_loss_test_est[:-1],  color = colors[1],alpha=0.2)
    ax_fig1.fill_between(nn[:-1], loss_test_est2[:-1] - sd_loss_test_est2[:-1], loss_test_est2[:-1] + sd_loss_test_est2[:-1],  color = colors[0],alpha=0.2)
    ax_fig1.fill_between(nn[:-1], loss_test_rand[:-1] - sd_loss_test_rand[:-1], loss_test_rand[:-1] + sd_loss_test_rand[:-1],  color = colors[3],alpha=0.2)
        
    
    
    
    mss = 4
    fs = "large"
    axin = inset_axes(ax_fig1, width="40%", height="15%", loc='lower left'
                      , bbox_to_anchor=(.15, .29, 1,1), bbox_transform=ax_fig1.transAxes)
    
    axin.plot(nn[:-1],loss_train[:-1],"-o",mec='black',label='exact', alpha=0.8, c = colors[6],markersize=mss)
    axin.plot(nn[:-1],loss_train_est[:-1],"-D",mec='black',label='est. (SWAP)', alpha=0.8, c = colors[1],markersize=mss)
    axin.plot(nn[:-1],loss_train_est2[:-1],"-s",mec='black',label='est. (overlap)', alpha=0.8, c = colors[0],markersize=mss)
    axin.plot(nn[:-1],loss_train_rand[:-1],"-^",mec='black',label='random', alpha=0.8, c = colors[3],markersize=mss)
    axin.minorticks_off()
    axin.set_ylabel(r'training loss',fontsize=fs)
    axin.set_xlabel(r'$N_s$',fontsize=fs)
    axin.tick_params(axis='y', labelsize=fs,direction="in")
    axin.tick_params(axis='x', labelsize=fs,direction="in")
    axin.set_ylim(-1,1)
    
    
    
    fig1.savefig('fig3-effect-expo-gen.pdf', bbox_inches='tight')#, transparent=True)


###############################################################################
###################### Figure 6 : Expressivity-induced concentration
###############################################################################

######## define a new setting and load some new data

layers = [2, 4, 8, 16, 32, 64, 128, 256]
layers_mnist = [2, 4, 8, 16, 25, 50, 75, 100, 150, 200]

qubits =  [2,4,6,8,10,12]
qubits_mnist =  [2,4,6,8,10,12,14]
qubits_ta = [2,6,10,12,14,16,18] 


kers = np.load('kernel-val.npy')
kers_mnist = np.load('kernel-val-mnist.npy')
pqk_mnist = np.load('kernel_proj_HEE_mnistdata_40sample.npy')
ta_rand = np.load('ta_test_random_HEE.npy')

var_kers = np.var(kers,axis=2)
var_kers_mnist = np.var(kers_mnist,axis=2)
var_pqk_mnist = np.var(pqk_mnist,axis=2)
var_ta = np.var(ta_rand,axis=2)[:,0]



colors = ['#7F8EC9','#4056A1','#075C2F','#7D8238','#453F3F','#692411','#D79922','#F13C20','black']

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

markers = ['-o','-s','-^','-D','-P']

colors3 = ["turquoise","firebrick"]

###############################################################################

colors1 = []
for i in range(len(layers_mnist[:8])):
    #shade = 0.85 - 0.05*i
    shade = 0.17 + 0.15*i
    #print(shade)
    colors1.append(plt.cm.OrRd(shade))

colors2 = []
for i in range(len(layers_mnist[:8])):
    #shade = 0.85 - 0.05*i
    shade = 0.17 + 0.15*i
    #print(shade)
    colors2.append(plt.cm.PuBu(shade))
    

fig1 = plt.figure(constrained_layout=True,figsize=(4.5,4.5))
fig1.set_constrained_layout_pads(w_pad=2./72., h_pad=2./72., hspace=0., wspace=0.)


gs1 = GridSpec(2,4,figure=fig1)
ax_a = fig1.add_subplot(gs1[0,0:])
ax_b = fig1.add_subplot(gs1[1,0:])

idx1 = 8

for i in range(len(layers_mnist[:idx1])):
    if i < 8:
        ax_a.plot(qubits_mnist[:-1],var_kers_mnist[:-1,i],markers[0], color=colors[i], label=layers_mnist[i])#,color=colors[0])
ax_a.set_yscale('log')
ax_a.set_ylim([3e-08, 9e-1])
ax_a.tick_params(axis='y', labelsize="large",direction="in")
ax_a.tick_params(axis='x', labelsize="large",direction="in")
ax_a.set_xticklabels([])
ax_a.minorticks_off()
ax_a.set_ylabel(r'${\rm Var}_{x}[\kappa^{FQ}]$',fontsize="x-large")
ax_a.legend(numpoints=1,ncol=2,loc=3,framealpha=1,edgecolor="black",fancybox=False, fontsize="small", title_fontsize='large')


for i in range(len(layers_mnist[:idx1])):
    if i < 8:
        ax_b.plot(qubits,var_pqk_mnist[:,i],markers[1], color=colors[i],label=layers_mnist[i])#, label=labels[0],color=colors[0])
ax_b.set_yscale('log')
ax_b.set_ylim([3e-08, 9e-1])
#ax_a.set_xticks(np.arange(2, 16, step=2))
#ax_a.set_xticklabels(np.arange(2, 16, step=2),fontsize="large")
ax_b.tick_params(axis='y', labelsize="large",direction="in")
ax_b.tick_params(axis='x', labelsize="large",direction="in")
ax_b.minorticks_off()

ax_b.set_xlabel(r'$n$',fontsize="x-large")
ax_b.set_ylabel(r'${\rm Var}_{x}[\kappa^{PQ}]$',fontsize="x-large")
ax_b.legend(numpoints=1,ncol=2,loc=3,framealpha=1,edgecolor="black",fancybox=False, fontsize="small", title_fontsize='large')

fig1.savefig('fig6-expressibility-concentration.pdf', bbox_inches='tight', transparent=True)
plt.show()

###############################################################################
###################### Figure 7 : Globality-induced concentration
###############################################################################

colors2 = [colors[5],colors[6],colors[7],colors[0],colors[1],colors[2],colors[3],colors[4]]
idx_stop = 5
markers = ['-s','-D','-P']

ker_rx = np.load('kernel_TPE_Rx_randomdata_60sample.npy')[:,0,:]
ker_ry = np.load('kernel_TPE_Ry_randomdata_60sample.npy')[:,0,:]
ker_rz = np.load('kernel_TPE_Rz_randomdata_60sample.npy')[:,0,:]

var_rx = np.var(ker_rx, axis=1)
var_ry = np.var(ker_ry, axis=1)
var_rz = np.var(ker_rz, axis=1)

var_r = [var_rx,var_ry,var_rz]

labels = [r'$R_x$',r'$R_y$',r'$H R_z$']

fig2 = plt.figure(constrained_layout=True,figsize=(4.5,2.25))
fig2.set_constrained_layout_pads(w_pad=2./72., h_pad=2./72., hspace=0., wspace=0.)

ax_fig2 = fig2.add_subplot()

for i in range(len(var_r)):
    ax_fig2.plot(qubits, var_r[i],markers[i],color=colors2[i],label=labels[i])
    
for i in range(len(layers[:idx_stop])):
    ax_fig2.plot(qubits, var_kers[:,i],'-o',color=colors2[i+3])#,label=layers[i])
ax_fig2.set_yscale('log')

ax_fig2.legend(numpoints=1,ncol=1,loc=3,framealpha=1,edgecolor="black",fancybox=False, fontsize="small", title_fontsize='large')
ax_fig2.tick_params(axis='y', labelsize="large",direction="in")
ax_fig2.tick_params(axis='x', labelsize="large",direction="in")
ax_fig2.minorticks_off()
ax_fig2.set_ylabel(r'${\rm Var}_{x}[\kappa^{FQ}]$',fontsize="x-large")
ax_fig2.set_xlabel(r'$n$',fontsize="x-large")

fig2.savefig('fig7-global-measurement-concentration.pdf', bbox_inches='tight', transparent=True)
plt.show()

###############################################################################
###################### Figure 8 : Noise-induced concentration
###############################################################################

noise = [0.01,0.025,0.05,0.075,0.10]
qubit_tot = [6] #6
sample = 60
num_pair = int(sample*(sample-1)/2)
layer_tot = [2,4,6,8,10,12,14,16,18,20,25,30] 

kernel_tot = np.load('kernel_noise_HEE_mnistdata_60sample.npy')

con = np.zeros((len(noise),len(qubit_tot),len(layer_tot)))

mix = np.zeros((len(qubit_tot),len(layer_tot),num_pair))

for i in range(len(qubit_tot)):
    mix[i] = mix[i] + 1/(2**qubit_tot[i])
    

con_ker = np.copy(kernel_tot)
for i in range(len(noise)):
    con_ker[i] = np.abs(kernel_tot[i] - mix)


con = np.mean(con_ker, axis=3)

qq = 1 - np.array(noise)

######################################################################### 

fig5 = plt.figure(constrained_layout=True,figsize=(4.5,4.5))
fig5.set_constrained_layout_pads(w_pad=2./72., h_pad=2./72., hspace=0., wspace=0.)


gs1 = GridSpec(2,4,figure=fig5)
ax_a = fig5.add_subplot(gs1[0,0:])
ax_b = fig5.add_subplot(gs1[1,0:])


con_noise_pqk = np.load('pqk_n8_noise.npy')
con_noise_fqk = np.load('fqk_n8_noise.npy')
layer_tot = [2,4,6,8,10,12,14,16,18,20,25] 

    
colors4 = [colors[0],colors[1],colors[2],colors[3],colors[6],colors[4],colors[6],colors[7]]


for i in range(len(noise)):
    ax_a.plot(layer_tot[:-1], con_noise_fqk[i,:-1],'-P',color=colors4[i],label=qq[i])
    ax_b.plot(layer_tot[:-1], con_noise_pqk[i,:-1],'-D',color=colors4[i],label=qq[i])
    
ax_a.set_yscale('log')
ax_a.set_xticklabels([])
ax_a.set_ylim([1e-16, 8e-1])
#ax_fig4.set_xticks(np.arange(2, 20, step=2))
ax_a.tick_params(axis='y', labelsize="large",direction="in")
ax_a.tick_params(axis='x', labelsize="large",direction="in")
ax_a.minorticks_off()
ax_a.set_ylabel(r'$\langle |\kappa^{FQ} - 1/2^n |\rangle_x$',fontsize="x-large")
#ax_a.set_xlabel(r'$L$',fontsize="x-large")
ax_a.legend(numpoints=1,ncol=1,loc=3,framealpha=1,edgecolor="black",fancybox=False, fontsize="small", title_fontsize='large')



ax_b.set_yscale('log')
ax_b.set_ylim([1e-16, 8e-1])
#ax_fig4.set_xticks(np.arange(2, 20, step=2))
ax_b.tick_params(axis='y', labelsize="large",direction="in")
ax_b.tick_params(axis='x', labelsize="large",direction="in")
ax_b.minorticks_off()
ax_b.set_ylabel(r'$\langle  |\kappa^{PQ} - 1|\rangle_x$',fontsize="x-large")
ax_b.set_xlabel(r'$L$',fontsize="x-large")
ax_b.legend(numpoints=1,ncol=1,loc=3,framealpha=1,edgecolor="black",fancybox=False, fontsize="small", title_fontsize='large')



fig5.savefig('fig8-noise-concentration.pdf', bbox_inches='tight', transparent=True)
plt.show()

###############################################################################
###################### Figure 9 : Noise-induced concentration
###############################################################################

fig3 = plt.figure(constrained_layout=True,figsize=(4.5,2.25))
fig3.set_constrained_layout_pads(w_pad=2./72., h_pad=2./72., hspace=0., wspace=0.)

ax_fig3 = fig3.add_subplot()
ax_fig3.plot(qubits_ta, var_ta,markers[1],color=colors[0])
ax_fig3.set_yscale('log')
ax_fig3.set_xticks(np.arange(2, 20, step=2))
ax_fig3.tick_params(axis='y', labelsize="large",direction="in")
ax_fig3.tick_params(axis='x', labelsize="large",direction="in")
ax_fig3.minorticks_off()
ax_fig3.set_ylabel(r'${\rm Var}_{\theta}[{\rm TA}]$',fontsize="x-large")
ax_fig3.set_xlabel(r'$n$',fontsize="x-large")
fig3.savefig('fig9-ta-kernel.pdf', bbox_inches='tight', transparent=True)

plt.show()

