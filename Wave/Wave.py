#!/usr/bin/env python
# coding: utf-8

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# experiment setup
lx=0
rx=1
lt=0
rt=1


# Define the exact solution and its derivatives
def u_func(X, a = 0.5, c = 2):
    """
    :param x: X = (x, t)
    """
    x = X[:,0:1]
    t = X[:,1:2]
    out = np.sin(np.pi * x) * np.cos(c * np.pi * t) + \
            a * np.sin(2 * c * np.pi* x) * np.cos(4 * c  * np.pi * t)
    
    return out

def u_t(X, a = 0.5, c = 2): #X = (x, t)
    x = X[:,0:1]
    t = X[:,1:2]
    u_t = -  c * np.pi * np.sin(np.pi * x) * np.sin(c * np.pi * t) - \
            a * 4 * c * np.pi * np.sin(2 * c * np.pi* x) * np.sin(4 * c * np.pi * t)
    return u_t


def u_tt(X, a = 0.5, c = 2): #X = (x, t)
    x = X[:,0:1]
    t = X[:,1:2]
    u_tt = -(c * np.pi)**2 * np.sin( np.pi * x) * np.cos(c * np.pi * t) - \
            a * (4 * c * np.pi)**2 *  np.sin(2 * c * np.pi* x) * np.cos(4 * c * np.pi * t)
    return u_tt


def u_xx(X, a = 0.5, c = 2): #X = (x, t)
    x = X[:,0:1]
    t = X[:,1:2]
    u_xx = - np.pi**2 * np.sin( np.pi * x) * np.cos(c * np.pi * t) - \
              a * (2 * c * np.pi)** 2 * np.sin(2 * c * np.pi* x) * np.cos(4 * c * np.pi * t)
    return  u_xx


def f_func(x, a = 0.5, c = 2):
    return u_tt(x, a, c) - c**2 * u_xx(x, a, c)



def sampler(num_r, num_b, num_0, lx, rx, lt, rt,  delta_N = 1001):
    # generate training data
    x = np.linspace(lx, rx, delta_N)
    t = np.linspace(lt, rt, delta_N)
    
    xx,tt = np.meshgrid(x,t)
    X = np.vstack([xx.ravel(), tt.ravel()]).T
    
    tb = np.linspace(lt, rt, num_b)
    # X boundaries
    lb   = lx*np.ones((tb.shape))
    rb   = rx*np.ones((tb.shape))
    Xlb  = np.vstack((lb,tb)).T
    Xrb  = np.vstack((rb,tb)).T
    UXlb = u_func(Xlb)
    UXrb = u_func(Xrb)
    xb = np.linspace(lx, rx, num_0)
    # T boundaries
    tlb   = lt*np.ones((xb.shape))
    Xic  = np.vstack((xb,tlb)).T
    Uic = u_func(Xic)
    
    # training tensors
    idxs = np.random.choice(xx.size, num_r, replace=False)
    X_train = torch.tensor(X[idxs], dtype=torch.float32, requires_grad=True,device=device)
    
    X_lb = torch.tensor(Xlb, dtype=torch.float32, device=device)
    X_rb = torch.tensor(Xrb, dtype=torch.float32, device=device)
    
    X_ic = torch.tensor(Xic, dtype=torch.float32, requires_grad=True,device=device)
    
    # compute mean and std of training data
    X_mean = torch.tensor(np.mean(np.concatenate([X[idxs], Xlb, Xrb, Xic], 0), axis=0, keepdims=True), dtype=torch.float32, device=device)
    
    X_std  = torch.tensor(np.std(np.concatenate([X[idxs], Xlb, Xrb, Xic], 0), axis=0, keepdims=True), dtype=torch.float32, device=device)
    
    U_X_lb = torch.tensor(UXlb, dtype=torch.float32, device=device).reshape(num_b,1)
    U_X_rb = torch.tensor(UXrb, dtype=torch.float32, device=device).reshape(num_b,1)
    
    U_ic = torch.tensor(Uic, dtype=torch.float32, requires_grad=True, device=device).reshape(num_0,1)
    
    return X_train, X_lb, X_rb, X_ic, U_X_lb, U_X_rb, U_ic, X_mean, X_std  


# computes pde residual
def Wave1D_res(uhat, data):
    x = data[:,0:1]
    y = data[:,1:2]
    du = grad(outputs=uhat, inputs=data, grad_outputs=torch.ones_like(uhat), create_graph=True)[0]
    dudx = du[:,0:1]
    dudxx = grad(outputs=dudx, inputs=data,grad_outputs=torch.ones_like(uhat), create_graph=True)[0][:,0:1]
    dudy = du[:,1:2]
    dudyy = grad(outputs=dudy, inputs=data,grad_outputs=torch.ones_like(uhat), create_graph=True)[0][:,1:2]
    
    residual = dudyy - 4*dudxx + 0*uhat  
    return residual



def Wave_res_u_t(uhat, data): #data = (x,t)
    poly = torch.ones_like(uhat)
    
    du = grad(outputs=uhat, inputs=data, 
              grad_outputs=torch.ones_like(uhat), create_graph=True)[0]
    
    dudt = du[:,1:2]
    return dudt + 0*uhat


lr = 1e-3  
mm         = 50 
i_print = 100
alpha_ann  = 0.1  

Adam_n_epochs   = 80000


num_r = 20000 
num_b = 1250  
num_0 = 2500  

layer_sizes = [2,500,500,500,500,500,1] 


extras=str(num_r)+ "+"+ str(num_b) + "+" + str(num_0)
guding_lr = False




if guding_lr:
    path_loc= './results/guding_lr_' + str(lr) + '_A-' + str(Adam_n_epochs) + '_rb0_' + extras + '_layer_' +str(len(layer_sizes[1:-1])) +'*' +str(layer_sizes[1])  
else:
    path_loc= './results/step_lr_' + str(lr) + '_A-' + str(Adam_n_epochs) + '_rb0_' + extras + '_layer_' +str(len(layer_sizes[1:-1])) +'*' +str(layer_sizes[1]) 


if not os.path.exists(path_loc):
    os.makedirs(path_loc)

print('guding_lr, lr: ', guding_lr, lr)
print('num_r, num_b, num_0: ', num_r, num_b, num_0)
print('layer_sizes: ', layer_sizes)



a = 0.5
c = 2

x = np.linspace(lx, rx, 1001)
t = np.linspace(lt, rt, 1001)
xx,tt = np.meshgrid(x,t)

X = np.vstack([xx.ravel(), tt.ravel()]).T
u_sol = u_func(X, a, c)



method_list = [0, 1, 2, 3, 'DB_PINN_mean', 'DB_PINN_std', 'DB_PINN_kurt']
#0: vanilla PINN (Equal Weighting); GW-PINN: 1: mean (max/avg); 2: std; 3: kurtosis;  


for i in range(7): 
    method = method_list[i]
    for j in range(1):
        
        print('i, j, method: ', i, j, method)
        save_loc = path_loc + '/method_' + str(method) + '/run_' + str(j) 
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)
        
        print("#######Training with#####\n",extras)
        
        X_train, X_lb, X_rb, X_ic, U_X_lb, U_X_rb, U_ic, X_mean, X_std= sampler(num_r, num_b, num_0, lx, rx, lt, rt)
        net = PINN(sizes=layer_sizes, mean=X_mean, std=X_std, activation=torch.nn.Tanh()).to(device)
        
        lambd_u      = torch.ones(1, device=device)
        lambd_ut       = torch.ones(1, device=device)
        lambd_u_all      = [];
        lambd_ut_all      = [];
        
        losses = []
        losses_reg  = [];
        losses_u  = [];
        losses_ut = [];
        l2_error = []
        
        N_l = 0
        
        params = [{'params': net.parameters(), 'lr': lr}]
        if guding_lr:
            optimizer = Adam(params)
        else:
            optimizer = Adam(params)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.9)  
        
        
        print("training with shape of residual points: ", X_train.size())
        print("training with shape of boundary points (*2): ", X_lb.size())
        print("training with shape of initial points: ", X_ic.size())
        
        start_time = time.time()
        for epoch in range(Adam_n_epochs): 
            
            uhat  = net(X_train)
            res   = Wave1D_res(uhat, X_train) 
            loss_reg = torch.mean((res)**2)

            X_u_all = torch.cat((X_lb, X_rb, X_ic), dim = 0)
            U_sol_all =  torch.cat((U_X_lb, U_X_rb, U_ic), dim = 0)
            pred_u = net(X_u_all) 
            loss_u = torch.mean((pred_u - U_sol_all)**2)
            
            pred_ic = net(X_ic) 
            gpreds= Wave_res_u_t(pred_ic, X_ic)
            loss_ut = torch.mean((gpreds)**2)  
            
            L_t = torch.stack((loss_reg, loss_u, loss_ut))
            
                                           
            
            with torch.no_grad():
                if epoch % mm == 0:
                    N_l += 1
                    stdr,kurtr=loss_grad_stats(loss_reg, net)
                    stdu,kurtu=loss_grad_stats(loss_u, net)
                    stdt,kurtt=loss_grad_stats(loss_ut, net)
                    
                    maxr,meanr=loss_grad_max_mean(loss_reg, net)
                    maxu,meanu=loss_grad_max_mean(loss_u, net,lambg=lambd_u) 
                    maxt,meant=loss_grad_max_mean(loss_ut, net,lambg=lambd_ut) 
                    
                    if epoch == 0:
                        lam_avg_u = torch.zeros(1, device=device)
                        lam_avg_ut = torch.zeros(1, device=device)
                        running_mean_L = torch.zeros(1, device=device)
                    
                    if method == 1:
                        # max/avg
                        lamb_hat = maxr/meanu
                        lambd_u    = (1-alpha_ann)*lambd_u + alpha_ann*lamb_hat 
                        lamb_hat = maxr/meant
                        lambd_ut     = (1-alpha_ann)*lambd_ut + alpha_ann*lamb_hat 
                        
                    elif method == 2:
                        # inverse dirichlet
                        lamb_hat = stdr/stdu
                        lambd_u     = (1-alpha_ann)*lambd_u + alpha_ann*lamb_hat
                        lamb_hat = stdr/stdt
                        lambd_ut     = (1-alpha_ann)*lambd_ut + alpha_ann*lamb_hat
                        
                        
                    elif method == 3:
                        # kurtosis based weighing
                        covr= stdr/kurtr
                        covu= stdu/kurtu
                        covt= stdt/kurtt
                        lamb_hat = covr/covu
                        lambd_u     = (1-alpha_ann)*lambd_u + alpha_ann*lamb_hat
                        lamb_hat = covr/covt
                        lambd_ut     = (1-alpha_ann)*lambd_ut + alpha_ann*lamb_hat
                        
                    elif method == 'DB_PINN_mean':  
                        
                        hat_all = maxr/meanu + maxr/meant
                        
                        mean_param = (1. - 1 / N_l)
                        running_mean_L = mean_param * running_mean_L + (1 - mean_param) * L_t.detach()
                        l_t_vector = L_t/running_mean_L
                        hat_u = hat_all* l_t_vector[1]/(l_t_vector[1] + l_t_vector[2])
                        hat_ut = hat_all* l_t_vector[2]/(l_t_vector[1] + l_t_vector[2])
                        lambd_u = lam_avg_u + 1/N_l*(hat_u - lam_avg_u)
                        lambd_ut = lam_avg_ut + 1/N_l*(hat_ut - lam_avg_ut)
                        lam_avg_u = lambd_u
                        lam_avg_ut = lambd_ut
                     
                    elif method == 'DB_PINN_std': 
                        hat_all = stdr/stdu + stdr/stdt
                        
                        mean_param = (1. - 1 / N_l)
                        running_mean_L = mean_param * running_mean_L + (1 - mean_param) * L_t.detach()
                        l_t_vector = L_t/running_mean_L
                        hat_u = hat_all* l_t_vector[1]/(l_t_vector[1] + l_t_vector[2])
                        hat_ut = hat_all* l_t_vector[2]/(l_t_vector[1] + l_t_vector[2])
                        lambd_u = lam_avg_u + 1/N_l*(hat_u - lam_avg_u)
                        lambd_ut = lam_avg_ut + 1/N_l*(hat_ut - lam_avg_ut)
                        lam_avg_u = lambd_u
                        lam_avg_ut = lambd_ut
                        
                        
                    elif method == 'DB_PINN_kurt': 
                        covr= stdr/kurtr
                        covu= stdu/kurtu
                        covt= stdt/kurtt
                        hat_all = covr/covu + covr/covt
                        
                        mean_param = (1. - 1 / N_l)
                        running_mean_L = mean_param * running_mean_L + (1 - mean_param) * L_t.detach()
                        l_t_vector = L_t/running_mean_L
                        hat_u = hat_all* l_t_vector[1]/(l_t_vector[1] + l_t_vector[2])
                        hat_ut = hat_all* l_t_vector[2]/(l_t_vector[1] + l_t_vector[2])
                        lambd_u = lam_avg_u + 1/N_l*(hat_u - lam_avg_u)
                        lambd_ut = lam_avg_ut + 1/N_l*(hat_ut - lam_avg_ut)
                        lam_avg_u = lambd_u
                        lam_avg_ut = lambd_ut
                    
                    else:
                        # equal weighting 
                        lambd_u = torch.ones(1, device=device)
                        lambd_ut = torch.ones(1, device=device)
            
            
            if lambd_u < 1 or torch.isnan(lambd_u) or torch.isinf(lambd_u)  or lambd_u > 1e8:
                lambd_u = torch.tensor(1.0, dtype=torch.float32, device=device)
            if lambd_ut < 1 or torch.isnan(lambd_ut) or torch.isinf(lambd_ut) or lambd_ut > 1e8:
                lambd_ut = torch.tensor(1.0, dtype=torch.float32, device=device)   
                            
            loss = loss_reg + lambd_u*loss_u + lambd_ut*loss_ut
            
            optimizer.zero_grad()
            loss.backward()
            if guding_lr:
                optimizer.step()
            else:
                optimizer.step()
                scheduler.step()

            if epoch%i_print==0:
                
                inp = torch.tensor(X, dtype=torch.float32, device=device)
                out = net(inp).cpu().data.numpy().reshape(u_sol.shape)
                tmp = np.linalg.norm(out.reshape(-1)-u_sol.reshape(-1))/np.linalg.norm(out.reshape(-1))
                
                l2_error.append(tmp)
                losses_reg.append(loss_reg.item())
                losses_u.append(loss_u.item())
                losses_ut.append(loss_ut.item())
                
                lambd_u_all.append(lambd_u.item())
                lambd_ut_all.append(lambd_ut.item())
                
                
                print("Adam optimizing: method={}  epoch {}/{}, loss={:.6f}, loss_r={:.6f}, loss_u={:.6f}, loss_ut={:.6f}, lambd_u={:.4f}, lambd_ut={:.4f}, lr={:.7f}, L2 error (%)={:.6f}".format(method, epoch, Adam_n_epochs, loss.item(), loss_reg.item(), loss_u.item(), loss_ut.item(), lambd_u.item(), lambd_ut.item(), optimizer.param_groups[0]['lr'], tmp*100)) 
                
                
        elapsed_time = time.time() - start_time
        inp = torch.tensor(X, dtype=torch.float32, device=device)
        out = net(inp).cpu().data.numpy().reshape(u_sol.shape)
        print("\n...Adam training...\n")
        print("Method: , j: ", str(method), j)
        print("Relative L2 error = {:e}\n".format(np.linalg.norm(out.reshape(-1)-u_sol.reshape(-1))/np.linalg.norm(u_sol.reshape(-1))))
        print("Mean absolute error = {:e}\n".format(np.mean(np.abs(out.reshape(-1)-u_sol.reshape(-1)))))
        print("\n.....\n")
        
        
        torch.save(net.state_dict(), os.path.join(save_loc, 'model.pth'))
        
        U_star = u_sol.reshape(xx.shape)
        U_pred = out.reshape(xx.shape)
        
        ##########plot results
        fig = plt.figure(1, figsize=(18, 5))
        fig_1 = plt.subplot(1, 3, 1)
        plt.pcolor(xx, tt, U_star, cmap='jet')
        plt.colorbar()
        plt.xlabel(r'$x$')
        plt.ylabel(r'$t$')
        plt.title('Exact $u(x)$')
        fig_2 = plt.subplot(1, 3, 2)
        plt.pcolor(xx, tt, U_pred, cmap='jet')
        plt.colorbar()
        plt.xlabel(r'$x$')
        plt.ylabel(r'$t$')
        plt.title('Predicted $u(x)$')
        fig_3 = plt.subplot(1, 3, 3)
        plt.pcolor(xx, tt, np.abs(U_star - U_pred), cmap='jet')
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
        ax.semilogy(losses_reg, label='$\mathcal{L}_{r}$')
        ax.semilogy(losses_u, label='$\mathcal{L}_{u}$')
        ax.semilogy(losses_ut , label='$\mathcal{L}_{ut}$')
        ax.set_yscale('log')
        ax.set_xlabel('iterations')
        ax.set_ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_loc, '2.loss.png'))
        plt.show()
        plt.close()
        
        
        ###########
        fig_3 = plt.figure(3)
        ax = fig_3.add_subplot(1, 1, 1)
        ax.semilogy(lambd_u_all, label='$\lambda_{u}$')
        ax.semilogy(lambd_ut_all, label='$\lambda_{ut}$')        
        ax.set_xlabel('iterations')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_loc,'3.learned_weights.png'))
        plt.show()
        plt.close()
        
        ###########
        fig_4 = plt.figure(4)
        ax = fig_4.add_subplot(1, 1, 1)
        ax.plot(l2_error)
        ax.set_xlabel('iterations')
        plt.tight_layout()
        plt.savefig(os.path.join(save_loc,'4.L2_error.png'))
        plt.show()
        plt.close()
        


