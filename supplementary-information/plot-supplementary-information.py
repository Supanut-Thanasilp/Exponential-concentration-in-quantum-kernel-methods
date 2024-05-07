"""

This is to plot supplementary figures in the supplementary information. Data have to already be stored in the folder. 

"""

# import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


plt.rc('text', usetex=True)
plt.rc('font', family='serif')

colors = ['#7F8EC9','#4056A1','#075C2F','#7D8238','#453F3F','#692411','#D79922','#F13C20','black']
markers = ['-o','-s','-^','-D','-P','-p','-*']


###############################################################################
###################### Supplementary Figure 4 : Effect of exponential concentration on Gram matrix with fidelity kernel
###############################################################################

supfig4_plot = True

if supfig4_plot:
    c_pattern = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    
    qubit_tot = np.array([5,7,10,15,20,30,40])
    num_qubit = qubit_tot
    dim_qubit = 2**qubit_tot
    nq = len(qubit_tot)
    ntrain = 25
    npp = int(ntrain*(ntrain-1)/2)
    
    shot_budgets = np.array([     10,      25,      50,      75,     100,     250,     500,
               750,    1000,    2500,    5000,    7500,   10000,   25000,
             50000,   75000,  100000,  250000,  500000,  750000, 1000000,
           1500000, 2000000])
    
    nb = len(shot_budgets)
    
    
    ### Overlap
    rel_error = np.load('fig_fidel_overlap_rel_error.npy')
    
    
    count_acc = np.zeros((nq,nb))
    count_zer = np.zeros((nq,nb))
    
    ratio_acc = np.zeros((nq,nb))
    ratio_zer = np.zeros((nq,nb))
    
    pbar = 1e-1
    
    
    for i in range(nq):
        for j in range(nb):
            count_acc[i][j] = (rel_error[i,:,j]<pbar).sum() 
            ratio_acc[i][j] = count_acc[i][j]/npp
            
            count_zer[i][j] = npp - np.count_nonzero(rel_error[i,:,j] - 1)
            ratio_zer[i][j] = count_zer[i][j]/npp
    
    ### SWAP
    ratio_p_suc = np.load('new_sup_fig_ratio_suc.npy')
    
    
    
    
    ### count the success ratio
    
    
    
    fig6 = plt.figure(constrained_layout=True,figsize=(6,4))
    fig6.set_constrained_layout_pads(w_pad=2./72., h_pad=2./72., hspace=0., wspace=0.)
    
    
    gs1 = GridSpec(4,4,figure=fig6)
    
    ax_a = fig6.add_subplot(gs1[:,:2])
    ax_b = fig6.add_subplot(gs1[:,2:])
    
    
    for i in range(nq):    
        if shot_budgets[0] < dim_qubit[i] < shot_budgets[-1]:
            ax_a.axvline(x=dim_qubit[i], linestyle='--', c = colors[i], alpha=0.9)
    
    for i in range(nq):
        ax_a.semilogx(shot_budgets, ratio_zer[i], markers[i], c=colors[i],label=num_qubit[i],mec='black', alpha=0.8)
        
    
    ax_a.tick_params(axis='y', labelsize="large",direction="in")
    ax_a.tick_params(axis='x', labelsize="large",direction="in")
    #ax_a.minorticks_off()
    ax_a.set_ylabel(r'zero ratio',fontsize="x-large")
    ax_a.set_xlabel(r'Number of shots, $N$',fontsize="x-large")
    #ax_a.legend(numpoints=1,ncol=1,loc='lower left',framealpha=1,edgecolor="black",fancybox=False, fontsize="medium", title_fontsize='large')
    
    
    
    for i in range(nq):    
        if shot_budgets[0] < dim_qubit[i] < shot_budgets[-1]:
            ax_b.axvline(x=dim_qubit[i], linestyle='--', c = colors[i], alpha=0.9)
    
    for i in range(nq):
        ax_b.semilogx(shot_budgets, ratio_p_suc[i], markers[i], c=colors[i],label=num_qubit[i],mec='black', alpha=0.8)
        
    
    ax_b.tick_params(axis='y', labelsize="large",direction="in")
    ax_b.tick_params(axis='x', labelsize="large",direction="in")
    #ax_a.minorticks_off()
    ax_b.set_ylabel(r'success ratio',fontsize="x-large")
    ax_b.set_xlabel(r'Number of shots, $N$',fontsize="x-large")
    ax_b.legend(numpoints=1,ncol=2,loc='upper left',framealpha=1,edgecolor="black",fancybox=False, fontsize="medium", title_fontsize='large')
    
    
    fig6.savefig('supfig4-effect-expo-gram.pdf', bbox_inches='tight', transparent=True)



###############################################################################
###################### Supplementary Figure 5 : Effect of exponential concentration on Gram matrix with projected kernel
###############################################################################


supfig5_plot = True

if supfig5_plot:
    c_pattern = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    colors = ['#7F8EC9','#4056A1','#075C2F','#7D8238','#453F3F','#692411','#D79922','#F13C20','black']
    markers = ['-o','-s','-^','-D','-P','-p']
    
    
    shot_budgets = np.load('new_sup_fig_shots.npy')
    
    num_qubit = np.array([ 7,  8,  9, 10, 11, 12])
    shot_budgets = np.array([    10,     25,     50,     75,    100,    250,    500,    750,
             1000,   2500,   5000,   7500,  10000,  25000,  50000,  75000,
           100000])
    
    nq = len(num_qubit)
    nb = len(shot_budgets)
    dim_qubit = 2**num_qubit
    
    
    ratio_p_suc = np.load('ratio_suc_swap.npy')
    ratio_p_suc2 = np.load('ratio_suc_tomo.npy')
    
    ### count the success ratio
    
    
    
    fig3 = plt.figure(constrained_layout=True,figsize=(6,4))
    fig3.set_constrained_layout_pads(w_pad=2./72., h_pad=2./72., hspace=0., wspace=0.)
    
    
    gs1 = GridSpec(4,4,figure=fig3)
    # ax_a = fig1.add_subplot(gs1[0,0:])
    # ax_b = fig1.add_subplot(gs1[1,0:])
    
    ax_a = fig3.add_subplot(gs1[:,:2])
    ax_b = fig3.add_subplot(gs1[:,2:])
    
    
    for i in range(nq):    
        if shot_budgets[0] < dim_qubit[i] < shot_budgets[-1]:
            ax_a.axvline(x=dim_qubit[i], linestyle='--', c = colors[i], alpha=0.9)
    
    for i in range(nq):
        ax_a.semilogx(shot_budgets, ratio_p_suc[i], markers[i], c=colors[i],label=num_qubit[i],mec='black', alpha=0.8)
        
    
    ax_a.tick_params(axis='y', labelsize="large",direction="in")
    ax_a.tick_params(axis='x', labelsize="large",direction="in")
    #ax_a.minorticks_off()
    ax_a.set_ylabel(r'success ratio',fontsize="x-large")
    ax_a.set_xlabel(r'Number of shots, $N$',fontsize="x-large")
    ax_a.legend(numpoints=1,ncol=2,loc=2,framealpha=1,edgecolor="black",fancybox=False, fontsize="medium", title_fontsize='large')
    
    
    
    for i in range(nq):    
        if shot_budgets[0] < dim_qubit[i] < shot_budgets[-1]:
            ax_b.axvline(x=dim_qubit[i], linestyle='--', c = colors[i], alpha=0.9)
    
    for i in range(nq):
        ax_b.semilogx(shot_budgets, ratio_p_suc2[i], markers[i], c=colors[i],label=num_qubit[i],mec='black', alpha=0.8)
        
    
    ax_b.tick_params(axis='y', labelsize="large",direction="in")
    ax_b.tick_params(axis='x', labelsize="large",direction="in")
    #ax_a.minorticks_off()
    ax_b.set_ylabel(r'success ratio',fontsize="x-large")
    ax_b.set_xlabel(r'Number of shots, $N$',fontsize="x-large")
    #ax_b.legend(numpoints=1,ncol=2,loc=2,framealpha=1,edgecolor="black",fancybox=False, fontsize="medium", title_fontsize='large')
    
    
    fig3.savefig('supfig5-proj-pval.pdf', bbox_inches='tight', transparent=True)


###############################################################################
###################### Supplementary Figure 6 :  Generalzation as a function of training data (projected kernel)
###############################################################################

supfig6_plot = True
if supfig6_plot:
    
    nn = np.array([ 10,  15,  20,  25,  30,  35,  40,  45,  50,  55,  60,  65,  70,
            75,  80,  85,  90,  95, 100, 105, 110, 115, 120, 125, 130, 135,
           140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195])
    
    
    
    ###############################################################################
    
    loss_test = np.load('sup_fig_gen_pqk_loss_test.npy')
    loss_test_est =  np.load('sup_fig_gen_pqk_loss_test_est_swap.npy')
    loss_test_est2 = np.load('sup_fig_gen_pqk_loss_test_est_tomo.npy')
    loss_test_rand = np.load('sup_fig_gen_pqk_loss_test_rand_swap.npy')
    loss_test_rand2 = np.load('sup_fig_gen_pqk_loss_test_rand_tomo.npy')
    
    loss_train = np.load('sup_fig_gen_pqk_loss_train.npy')
    loss_train_est = np.load('sup_fig_gen_pqk_loss_train_est_swap.npy')
    loss_train_est2 = np.load('sup_fig_gen_pqk_loss_train_est_tomo.npy')
    loss_train_rand = np.load('sup_fig_gen_pqk_loss_train_rand_swap.npy')
    loss_train_rand2 = np.load('sup_fig_gen_pqk_loss_train_rand_tomo.npy')
    
    
    fig4 = plt.figure(constrained_layout=True) #,figsize=(4.5,2.25))
    fig4.set_constrained_layout_pads(w_pad=2./72., h_pad=2./72., hspace=0., wspace=0.)
    ax_fig4 = fig4.add_subplot()
    
    ax_fig4.plot(nn[:-1],loss_test[:-1]/loss_test[0],'-o',mec='black',label='exact', c = colors[6], alpha=0.9)
    ax_fig4.plot(nn[:-1],loss_test_est[:-1]/loss_test_est[0],"-D",mec='black',label='est(SWAP)', c = colors[1], alpha=0.9)
    ax_fig4.plot(nn[:-1],loss_test_est2[:-1]/loss_test_est2[0],"-s",mec='black',label='est(Tomo.)', c = colors[0], alpha=0.9)
    ax_fig4.plot(nn[:-1],loss_test_rand[:-1]/loss_test_rand[0],"-^",mec='black',label='rand(SWAP)', c = colors[3], alpha=0.9)
    ax_fig4.plot(nn[:-1],loss_test_rand2[:-1]/loss_test_rand2[0],"-^",mec='black',label='rand(Tomo.)', c = colors[4], alpha=0.9)
    ax_fig4.set_yscale('log')
    ax_fig4.tick_params(axis='y', labelsize="large",direction="in")
    ax_fig4.tick_params(axis='x', labelsize="large",direction="in")
    ax_fig4.minorticks_off()
    ax_fig4.set_ylabel(r'relative test loss, $\eta$',fontsize="x-large")
    ax_fig4.set_xlabel(r'Number of training data, $N_s$',fontsize="x-large")
    ax_fig4.legend(numpoints=1,ncol=2,loc='lower left',framealpha=0.8,edgecolor="black",fancybox=False, fontsize="small", title_fontsize='large')
    #ax_fig1.legend(numpoints=1,ncol=1,loc='upper left', bbox_to_anchor=(1, 0.5),framealpha=1,edgecolor="black",fancybox=False, fontsize="medium", title_fontsize='large')
    
    
    
    mss = 4
    
    axin = inset_axes(ax_fig4, width="40%", height="15%", loc='lower left'
                  , bbox_to_anchor=(.55, .3, 1, 1), bbox_transform=ax_fig4.transAxes)
    
    axin.plot(nn[:-1],loss_train[:-1],"-o",mec='black',label='exact', alpha=0.8, c = colors[6],markersize=mss)
    axin.plot(nn[:-1],loss_train_est[:-1],"-D",mec='black',label='est(SWAP)', alpha=0.8, c = colors[1],markersize=mss)
    axin.plot(nn[:-1],loss_train_est2[:-1],"-s",mec='black',label='est(Tomo.)', alpha=0.8, c = colors[0],markersize=mss)
    axin.plot(nn[:-1],loss_train_rand[:-1],"-^",mec='black',label='rand(SWAP)', alpha=0.8, c = colors[3],markersize=mss)
    axin.plot(nn[:-1],loss_train_rand2[:-1],"-^",mec='black',label='rand(Tomo.)', alpha=0.8, c = colors[3],markersize=mss)
    
    axin.minorticks_off()
    axin.set_ylabel(r'training loss',fontsize="medium")
    axin.set_xlabel(r'$N_s$',fontsize="medium")
    axin.tick_params(axis='y', labelsize="medium",direction="in")
    axin.tick_params(axis='x', labelsize="medium",direction="in")
    axin.set_ylim(-1,1)
    fig4.savefig('supfig6-effect-expo-gen-pqk.pdf', bbox_inches='tight', transparent=True)

