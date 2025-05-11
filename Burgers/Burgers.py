import os
import time
import numpy as np
import sys
sys.path.append("..")
from pinn import *
from grad_stats import *
import math
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import grad
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ExponentialLR, MultiStepLR
from scipy.interpolate import griddata
from pyDOE import lhs
import scipy.io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sampler(num_r, num_b, num_0):
    
    lb = np.array([-1.0])
    ub = np.array([1.0])
    
    data = scipy.io.loadmat('../data/burgers_shock.mat') #data released from SA-PINN
    
    t = data['t'].flatten()[:,None] 
    x = data['x'].flatten()[:,None] 
    Exact = data['usol']
    Exact_u = np.real(Exact) 
    
    #grab training points from domain
    idx_x = np.random.choice(x.shape[0], num_0, replace=False)
    x0 = x[idx_x,:]  
    u0 = Exact_u[idx_x,0:1]
    
    idx_t = np.random.choice(t.shape[0], num_b, replace=False)
    tb = t[idx_t,:]
    
    # Grab collocation points using latin hpyercube sampling
    X_f = lb + (ub-lb)*lhs(2, num_r)
    X_f[:,1:2] = np.abs(X_f[:,1:2])  
    
    X0 = np.concatenate((x0, np.abs(0*x0)), 1) 
    X_lb = np.concatenate((0*tb + lb[0], tb), 1) 
    X_ub = np.concatenate((0*tb + ub[0], tb), 1)
    
    
    #generate meshgrid
    X, T = np.meshgrid(x,t) 
    
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_sol = Exact_u.T.flatten()[:,None]  
    
    X_mean = torch.tensor(np.mean(np.concatenate([X_f, X_lb, X_ub, X0], 0), axis=0, keepdims=True), dtype=torch.float32, device=device)
    
    X_std  = torch.tensor(np.std(np.concatenate([X_f, X_lb, X_ub, X0], 0), axis=0, keepdims=True), dtype=torch.float32, device=device)
    
    X_train = torch.tensor(X_f, dtype=torch.float32, requires_grad=True,device=device)
    
    X_lb = torch.tensor(X_lb, dtype=torch.float32, device=device, requires_grad=True)
    X_rb = torch.tensor(X_ub, dtype=torch.float32, device=device, requires_grad=True)
    
    X_ic = torch.tensor(X0, dtype=torch.float32, device=device)
    
    # compute mean and std of training data
    
    
    U_ic = torch.tensor(u0, dtype=torch.float32, device=device)
    
    return X_train, X_lb, X_rb, X_ic, U_ic, X_mean, X_std, X, T, Exact_u, X_star, u_sol



###### computes pde residual
def Burgers_res(uhat, data, lam1, lam2):  #data: (x, t)
    x = data[:,0:1]
    t = data[:,1:2]
    
    du = grad(outputs=uhat, inputs=data, 
              grad_outputs=torch.ones_like(uhat), create_graph=True)[0]
    
    dudx = du[:,0:1]
    dudt = du[:,1:2]
    
    dudxx = grad(outputs=dudx, inputs=data, 
              grad_outputs=torch.ones_like(uhat), create_graph=True)[0][:,0:1]
                                                                                                                     
    residual = dudt + lam1*uhat*dudx - lam2*dudxx
    return residual




all_losses=[]
list_of_l2_Errors=[]
i_print = 10

lr = 1e-3  
lr_lbfgs = 0.1  

mm         = 10 
alpha_ann  = 0.1 

Adam_n_epochs   = 5000
LBFGS_n_epochs  = 5000

layer_sizes = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
num_0 = 100
num_b = 100   #actually N_b is 200 for lb and ub
num_r = 10000
num_u = 300
D_nu = 0.01/np.pi


guding_lr = True
if guding_lr:
    path_loc= './results/guding_lr_%s_rb0u_%s_%s_%s_%s_%s' % (lr, num_r, num_b, num_0, num_u, layer_sizes) + '/A-' + str(Adam_n_epochs) + 'L-' + str(LBFGS_n_epochs) + '_lr_lbfgs_' + str(lr_lbfgs)
else:
    path_loc= './results/step_lr_%s_rb0u_%s_%s_%s_%s_%s' % (lr, num_r, num_b, num_0, num_u, layer_sizes) + '/A-' + str(Adam_n_epochs) +  'L-' + str(LBFGS_n_epochs) + '_lr_lbfgs_' + str(lr_lbfgs)


print('guding_lr, lr: ', guding_lr, lr)
print('num_r, num_b, num_0, num_u: ', num_r, num_b, num_0, num_u)
print('layer_sizes: ', layer_sizes)

if not os.path.exists(path_loc):
    os.makedirs(path_loc)



method_list = [0, 1, 2, 3, 'DB_PINN_mean', 'DB_PINN_std', 'DB_PINN_kurt']
#0: vanilla PINN (Equal Weighting); GW-PINN: 1: mean (max/avg); 2: std; 3: kurtosis;  

for i in range(7): 
    method = method_list[i]
    for j in range(1):

        lambda1=2.0
        lambda2=0.2

        lambda1 = torch.tensor([lambda1], requires_grad=True).float().to(device) 
        lambda2 = torch.tensor([lambda2], requires_grad=True).float().to(device)

        lambda1 = nn.Parameter(lambda1)
        lambda2 = nn.Parameter(lambda2)

        print('i, j, method: ', i, j, method)
        save_loc = path_loc + '/method_' + str(method) + '/run_' + str(j) 
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)
        
        extras=str(num_r)+ "+"+ str(num_b) + "+" + str(num_0)
        print("#######Training with#####\n",extras)
        
        X_train, X_lb, X_rb, X_ic, U_ic, X_mean, X_std, X, T, Exact_u, X_star, u_sol = sampler(num_r, num_b, num_0)
        id_u = np.random.choice(X_star.shape[0], num_u, replace=False)
        X_train_Nu = torch.tensor(X_star[id_u,:], dtype=torch.float32, device=device)  
        U_train_Nu = torch.tensor(u_sol[id_u,:], dtype=torch.float32, device=device)  
        
        net = PINN(sizes=layer_sizes, mean=X_mean, std=X_std, activation=torch.nn.Tanh()).to(device)
        
        net.register_parameter('lambda1', lambda1) 
        net.register_parameter('lambda2', lambda2)
        for k,v in net.named_parameters():
            print(k,v)
        
        lambd_r       = torch.ones(1, device=device)
        lambd_bc       = torch.ones(1, device=device)
        lambd_ic       = torch.ones(1, device=device)
        lambd_d        = torch.ones(1, device=device)
        
        lambd_r_all       = [];
        lambd_bc_all      = [];
        lambd_ic_all      = [];
        lambd_d_all      = [];
        
        losses = []
        losses_initial  = [];
        losses_boundary  = [];
        losses_residual = [];
        losses_data = [];
        
        l2_error = []

        lambda1_all = []
        lambda2_all = []
        N_l = 0
        
        params = [{'params': net.parameters(), 'lr': lr}] 
        milestones = [[10000,20000,30000]]
        
        if guding_lr:
            optimizer = Adam(params, betas=(0.99, 0.999)) 
        else:
            optimizer = Adam(params)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.9) 
        
        print("training with shape of residual points: ", X_train.size())
        print("training with shape of boundary points (*2): ", X_lb.size())
        print("training with shape of initial points: ", X_ic.size())
        
        start_time = time.time()
        for epoch in range(Adam_n_epochs): 
            
            uhat  = net(X_train)
            res   = Burgers_res(uhat, X_train, lambda1, lambda2)
            l_reg = torch.mean((res)**2)
            
            predl = net(X_lb)
            predr = net(X_rb)
            
            l_bc  = torch.mean((predl - 0)**2)
            l_bc += torch.mean((predr - 0)**2)                
            
            pred_ic = net(X_ic)
            l_ic = torch.mean((pred_ic - U_ic)**2)
            
            pred_u = net(X_train_Nu)
            l_data = torch.mean((pred_u - U_train_Nu)**2)
            
            L_t = torch.stack((l_reg, l_bc, l_ic, l_data))
            
            with torch.no_grad():
                if epoch % mm == 0:
                    N_l += 1
                    
                    stdr,kurtr=loss_grad_stats(l_reg, net)
                    stdb,kurtb=loss_grad_stats(l_bc, net)
                    stdi,kurti=loss_grad_stats(l_ic, net)
                    stdd,kurtd=loss_grad_stats(l_data, net)
                    
                    maxr,meanr=loss_grad_max_mean(l_reg, net)
                    maxb,meanb=loss_grad_max_mean(l_bc, net,lambg=lambd_bc)
                    maxi,meani=loss_grad_max_mean(l_ic, net,lambg=lambd_ic)
                    maxd,meand=loss_grad_max_mean(l_data, net,lambg=lambd_d)
                    
                    if epoch == 0:
                        lam_avg_bc = torch.zeros(1, device=device)
                        lam_avg_ic = torch.zeros(1, device=device)
                        lam_avg_d = torch.zeros(1, device=device)
                        running_mean_L = torch.zeros(1, device=device)
                    
                    if method == 1:
                        # max/avg
                        lamb_hat = maxr/meanb
                        lambd_bc     = (1-alpha_ann)*lambd_bc + alpha_ann*lamb_hat 
                        lamb_hat = maxr/meani
                        lambd_ic     = (1-alpha_ann)*lambd_ic + alpha_ann*lamb_hat 
                        lamb_hat = maxr/meand
                        lambd_d      = (1-alpha_ann)*lambd_d + alpha_ann*lamb_hat 
                        
                    elif method == 2:
                        # inverse dirichlet
                        lamb_hat = stdr/stdb
                        lambd_bc     = (1-alpha_ann)*lambd_bc + alpha_ann*lamb_hat
                        lamb_hat = stdr/stdi
                        lambd_ic     = (1-alpha_ann)*lambd_ic + alpha_ann*lamb_hat
                        lamb_hat = stdr/stdd
                        lambd_d     = (1-alpha_ann)*lambd_d + alpha_ann*lamb_hat
                        
                    elif method == 3:
                        # kurtosis based weighing
                        covr= stdr/kurtr
                        covb= stdb/kurtb
                        covi= stdi/kurti
                        covd= stdd/kurtd
                        
                        lamb_hat = covr/covb
                        lambd_bc     = (1-alpha_ann)*lambd_bc + alpha_ann*lamb_hat
                        lamb_hat = covr/covi
                        lambd_ic     = (1-alpha_ann)*lambd_ic + alpha_ann*lamb_hat
                        lamb_hat = covr/covd
                        lambd_d     = (1-alpha_ann)*lambd_d + alpha_ann*lamb_hat
                        
                    elif method == 'DB_PINN_mean':  
                        hat_all = maxr/meanb + maxr/meani + maxr/meand
                        
                        mean_param = (1. - 1 / N_l)
                        running_mean_L = mean_param * running_mean_L + (1 - mean_param) * L_t.detach()
                        l_t_vector = L_t/running_mean_L
                        hat_bc = hat_all* l_t_vector[1]/torch.sum(l_t_vector[1:])  
                        hat_ic = hat_all* l_t_vector[2]/torch.sum(l_t_vector[1:])  
                        hat_d = hat_all* l_t_vector[3]/torch.sum(l_t_vector[1:])  
                        
                        lambd_bc = lam_avg_bc + 1/N_l*(hat_bc - lam_avg_bc)
                        lambd_ic = lam_avg_ic + 1/N_l*(hat_ic - lam_avg_ic)
                        lambd_d =  lam_avg_d  + 1/N_l*(hat_d - lam_avg_d)
                        
                        lam_avg_bc = lambd_bc
                        lam_avg_ic = lambd_ic
                        lam_avg_d = lambd_d
                            
                    elif method == 'DB_PINN_std': 
                        hat_all = stdr/stdb + stdr/stdi + stdr/stdd
                        
                        mean_param = (1. - 1 / N_l)
                        running_mean_L = mean_param * running_mean_L + (1 - mean_param) * L_t.detach()
                        l_t_vector = L_t/running_mean_L
                        hat_bc = hat_all* l_t_vector[1]/torch.sum(l_t_vector[1:])  
                        hat_ic = hat_all* l_t_vector[2]/torch.sum(l_t_vector[1:])  
                        hat_d = hat_all* l_t_vector[3]/torch.sum(l_t_vector[1:])  
                        
                        lambd_bc = lam_avg_bc + 1/N_l*(hat_bc - lam_avg_bc)
                        lambd_ic = lam_avg_ic + 1/N_l*(hat_ic - lam_avg_ic)
                        lambd_d =  lam_avg_d  + 1/N_l*(hat_d - lam_avg_d)
                        
                        lam_avg_bc = lambd_bc
                        lam_avg_ic = lambd_ic
                        lam_avg_d = lambd_d
                    
                    elif method == 'DB_PINN_kurt':
                        covr= stdr/kurtr
                        covb= stdb/kurtb
                        covi= stdi/kurti
                        covd= stdd/kurtd
                        hat_all = covr/covb + covr/covi + covr/covd
                        
                        mean_param = (1. - 1 / N_l)
                        running_mean_L = mean_param * running_mean_L + (1 - mean_param) * L_t.detach()
                        l_t_vector = L_t/running_mean_L
                        hat_bc = hat_all* l_t_vector[1]/torch.sum(l_t_vector[1:])  
                        hat_ic = hat_all* l_t_vector[2]/torch.sum(l_t_vector[1:])  
                        hat_d = hat_all* l_t_vector[3]/torch.sum(l_t_vector[1:])  
                        
                        lambd_bc = lam_avg_bc + 1/N_l*(hat_bc - lam_avg_bc)
                        lambd_ic = lam_avg_ic + 1/N_l*(hat_ic - lam_avg_ic)
                        lambd_d =  lam_avg_d  + 1/N_l*(hat_d - lam_avg_d)
                        
                        lam_avg_bc = lambd_bc
                        lam_avg_ic = lambd_ic
                        lam_avg_d = lambd_d
                    
                    else:
                        # equal weighting 
                        lambd_bc = torch.ones(1, device=device)
                        lambd_ic = torch.ones(1, device=device)
                        lambd_d = torch.ones(1, device=device)
            

            loss = l_reg + lambd_bc.item()*l_bc + lambd_ic.item()*l_ic + lambd_d.item()*l_data
            
            if epoch%i_print==0:
                
                inp = torch.tensor(X_star, dtype=torch.float32, device=device, requires_grad=True)
                out = net(inp).cpu().data.numpy().reshape(u_sol.shape)
                tmp = np.linalg.norm(out.reshape(-1)-u_sol.reshape(-1))/np.linalg.norm(out.reshape(-1))
                
                l2_error.append(tmp)
                list_of_l2_Errors.append(tmp)
                all_losses.append(loss.item())
                
                losses_initial.append(l_ic.item())
                losses_boundary.append(l_bc.item())
                losses_residual.append(l_reg.item())
                losses_data.append(l_data.item())
                
                lambd_r_all.append(lambd_r.item())
                lambd_bc_all.append(lambd_bc.item())
                lambd_ic_all.append(lambd_ic.item())
                lambd_d_all.append(lambd_d.item())
                
                print("Adam method={} epoch {}/{}, loss={:.4f}, loss_r={:.6f}, loss_bc={:.6f}, loss_ic={:.6f}, loss_d={:.6f}, lam_r={:.4f}, lam_bc={:.4f}, lam_ic={:.4f}, lam_d={:.4f}, lr={:.5f}, l2_error(%)={:.3f}".format(method, epoch+1, Adam_n_epochs, loss.item(), l_reg.item(), l_bc.item(), l_ic.item(), l_data.item(), lambd_r.item(), lambd_bc.item(), lambd_ic.item(), lambd_d.item(), optimizer.param_groups[0]['lr'], tmp*100))
                
                print('𝜆_real = [1.0,  %.5f], 𝜆_PINN = [%.5f,  %.5f]'%(D_nu, lambda1.item(), lambda2.item()))
                lambda1_all.append(lambda1.item())
                lambda2_all.append(lambda2.item())
            
            optimizer.zero_grad()
            loss.backward()
            if guding_lr:
                optimizer.step()
            else:
                optimizer.step()
                scheduler.step()
        
        elapsed_time = time.time() - start_time
        print('Adam training time = ', elapsed_time)
        inp = torch.tensor(X_star, dtype=torch.float32, device=device, requires_grad=True )
        out = net(inp)
        out = out.cpu().data.numpy().reshape(u_sol.shape)
        
        print("\n...Adam training...\n")
        print("Method: , j: ", str(method), j)
        print("pred rel. l2-error = {:e}\n".format(np.linalg.norm(out.reshape(-1)-u_sol.reshape(-1))/np.linalg.norm(u_sol.reshape(-1))))
        print("pred abs. error = {:e}\n".format(np.mean(np.abs(out.reshape(-1)-u_sol.reshape(-1)))))
        print("\n.....\n")
        
        
        ##############start LBFGS training............#################
        optimizer_lbfgs = torch.optim.LBFGS(net.parameters(), lr_lbfgs, 
                              max_iter = LBFGS_n_epochs, 
                              max_eval = None, 
                              tolerance_grad = 1e-23,   
                              tolerance_change = 1e-23,  
                              history_size = 100, 
                              line_search_fn = 'strong_wolfe')
        loss_lbfgs_all = []
        loss_lbfgs_bc = []
        loss_lbfgs_r = []
        loss_lbfgs_ic = []
        loss_lbfgs_d = []
        
        l2_error_lbfgs = []
        lambda1_all_lbfgs = []
        lambda2_all_lbfgs = []
        N_l = 0
        epoch_l = 0
        def loss_pinn():
            global epoch_l
            global N_l
            global lambda1
            global lambda2
            
            optimizer_lbfgs.zero_grad()
            uhat  = net(X_train)
            res   = Burgers_res(uhat, X_train, lambda1, lambda2)
            l_reg = torch.mean((res)**2)
            
            predl = net(X_lb)
            predr = net(X_rb)
            
            l_bc  = torch.mean((predl - 0)**2)
            l_bc  += torch.mean((predr - 0)**2)                                
            
            pred_ic = net(X_ic)
            
            l_ic = torch.mean((pred_ic - U_ic)**2)
            
            pred_u = net(X_train_Nu)
            l_data = torch.mean((pred_u - U_train_Nu)**2)
            
            loss = l_reg + lambd_bc.item()*l_bc + lambd_ic.item()*l_ic + lambd_d.item()*l_data
            loss.backward()
            if epoch_l%i_print==0:
                
                inp = torch.tensor(X_star, dtype=torch.float32, device=device, requires_grad=True)
                out = net(inp).cpu().data.numpy().reshape(u_sol.shape)
                tmp = np.linalg.norm(out.reshape(-1)-u_sol.reshape(-1))/np.linalg.norm(out.reshape(-1))
                
                l2_error_lbfgs.append(tmp)
                loss_lbfgs_all.append(loss.item())
                
                loss_lbfgs_r.append(l_reg.item())   
                loss_lbfgs_bc.append(l_bc.item())
                loss_lbfgs_ic.append(l_ic.item())
                loss_lbfgs_d.append(l_data.item())
                
                print("LBFGS method={}, N_l {}, epoch {}/{}, loss={:.4f}, loss_r={:.6f}, loss_bc={:.6f}, loss_ic={:.6f}, lam_r={:.4f}, lam_bc={:.4f}, lam_ic={:.4f}, lr_lbfgs={:.3f}, l2_error(%)={:.3f}".format(method, N_l, epoch_l, LBFGS_n_epochs, loss.item(), l_reg.item(), l_bc.item(), l_ic.item(), lambd_r.item(), lambd_bc.item(), lambd_ic.item(), lr_lbfgs, tmp*100))
                print('𝜆_real = [1.0,  %.5f], 𝜆_PINN = [%.5f,  %.5f]'%(D_nu, lambda1.item(), lambda2.item()))
                lambda1_all_lbfgs.append(lambda1.item())
                lambda2_all_lbfgs.append(lambda2.item())
            
            epoch_l += 1
            return loss
        
        if LBFGS_n_epochs !=0:
            start_time = time.time() 
            
            optimizer_lbfgs.step(loss_pinn)
            elapsed_time = time.time() - start_time
            print('LBFGS training time = ',elapsed_time)

            inp = torch.tensor(X_star, dtype=torch.float32, device=device, requires_grad=True )
            out = net(inp)
            out = out.cpu().data.numpy().reshape(u_sol.shape)
            
            print("\n...LBFGS training...\n")
            print("Method: , j: ",method, j)
            print("pred rel. l2-error = {:e}\n".format(np.linalg.norm(out.reshape(-1)-u_sol.reshape(-1))/np.linalg.norm(out.reshape(-1))))
            print("pred abs. error = {:e}\n".format(np.mean(np.abs(out.reshape(-1)-u_sol.reshape(-1)))))
            print("\n.....\n")
            
        
        U_pred = griddata(X_star, out.flatten(), (X, T), method='cubic')
        
        ###########
        fig = plt.figure(1, figsize=(18, 5))
        fig_1 = plt.subplot(1, 3, 1)
        plt.pcolor(X, T, Exact_u.T, cmap='jet')
        plt.colorbar()
        plt.xlabel(r'$x$')
        plt.ylabel(r'$t$')
        plt.title('Exact $u(x)$')
        fig_2 = plt.subplot(1, 3, 2)
        plt.pcolor(X, T, U_pred, cmap='jet')
        plt.colorbar()
        plt.xlabel(r'$x$')
        plt.ylabel(r'$t$')
        plt.title('Predicted $u(x)$')
        fig_3 = plt.subplot(1, 3, 3)
        plt.pcolor(X, T, np.abs(Exact_u.T - U_pred), cmap='jet')
        plt.colorbar()
        plt.xlabel(r'$x$')
        plt.ylabel(r'$t$')
        plt.title('Absolute error')
        plt.tight_layout()
        plt.savefig(os.path.join(save_loc,'1.predictions.png'))
        plt.show()
        plt.close()

        ###########
        fig_2 = plt.figure(2)
        ax = fig_2.add_subplot(1, 1, 1)
        ax.plot(losses_residual, label='$\mathcal{L}_{r}$')
        ax.plot(losses_boundary, label='$\mathcal{L}_{bc}$')
        ax.plot(losses_initial, label='$\mathcal{L}_{ic}$') 
        ax.plot(losses_data, label='$\mathcal{L}_{d}$')         
        ax.set_yscale('log')
        ax.set_xlabel('iterations')
        ax.set_ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_loc, '2.loss.png'))
        plt.show()
        plt.close()
        
        
        fig_3 = plt.figure(3)
        ax = fig_3.add_subplot(1, 1, 1)
        ax.plot(lambd_bc_all, label='$\lambda_{bc}$')
        ax.plot(lambd_ic_all, label='$\lambda_{ic}$')
        ax.plot(lambd_d_all, label='$\lambda_{d}$')
        ax.set_xlabel('iterations')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_loc,'3.learned_weights.png'))
        plt.show()
        plt.close()
        
        fig_4 = plt.figure(4)
        ax = fig_4.add_subplot(1, 1, 1)
        ax.plot(l2_error)
        ax.set_xlabel('iterations')
        plt.tight_layout()
        plt.savefig(os.path.join(save_loc,'4.L2_error.png'))
        plt.show()
        plt.close()

        fig_5 = plt.figure(5)
        ax = fig_5.add_subplot(1, 1, 1)
        l = len(lambda1_all)
        plt.plot(range(l), lambda1_all, "r-")
        plt.plot(range(l), lambda2_all, "k-")
        plt.plot(range(l), np.ones(l) * 1, "r--")
        plt.plot(range(l), np.ones(l) * D_nu, "k--")
        plt.legend(["C1hat", "C2hat", "True C1", "True C2"], loc="right")
        plt.xlabel("Epoch")
        plt.tight_layout()
        plt.savefig(os.path.join(save_loc,'5.inverse_pred_param.png'))
        plt.show()
        plt.close()

################################### all plot
        fig_2 = plt.figure(2)
        ax = fig_2.add_subplot(1, 1, 1)
        ax.plot(losses_residual + loss_lbfgs_r, label='$\mathcal{L}_{r}$')
        ax.plot(losses_boundary + loss_lbfgs_bc, label='$\mathcal{L}_{bc}$')
        ax.plot(losses_initial + loss_lbfgs_ic, label='$\mathcal{L}_{ic}$') 
        ax.plot(losses_data + loss_lbfgs_d, label='$\mathcal{L}_{d}$')         
        ax.set_yscale('log')
        ax.set_xlabel('iterations')
        ax.set_ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_loc, 'all_loss.png'))
        plt.show()
        plt.close()

        fig_4 = plt.figure(4)
        ax = fig_4.add_subplot(1, 1, 1)
        ax.plot(l2_error + l2_error_lbfgs)
        ax.set_xlabel('iterations')
        plt.tight_layout()
        plt.savefig(os.path.join(save_loc,'all_L2_error.png'))
        plt.show()
        plt.close()
        
        fig_5 = plt.figure(5)
        ax = fig_5.add_subplot(1, 1, 1)
        l = len(lambda1_all + lambda1_all_lbfgs)
        plt.plot(range(l), lambda1_all + lambda1_all_lbfgs, "r-")
        plt.plot(range(l), lambda2_all + lambda2_all_lbfgs, "k-")
        plt.plot(range(l), np.ones(l) * 1, "r--")
        plt.plot(range(l), np.ones(l) * D_nu, "k--")
        plt.legend(["C1hat", "C2hat", "True C1", "True C2"], loc="right")
        plt.xlabel("Epoch")
        plt.tight_layout()
        plt.savefig(os.path.join(save_loc,'all_inverse_pred_param.png'))
        plt.show()
        plt.close()
        
