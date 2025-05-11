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
lt=0
rx=1
rt=1


def kg_equation(x,y): #(x,t)
    return x*np.cos(5*np.pi*y) + (x*y)**3

def u_func(x): #x = (x, t)
    return x[:, 0:1] * np.cos(5 * np.pi * x[:, 1:2]) + (x[:, 1:2] * x[:, 0:1])**3

def u_tt(x): #x = (x, t)
    return - 25 * np.pi**2 * x[:, 0:1] * np.cos(5 * np.pi * x[:, 1:2]) + 6 * x[:,1:2] * x[:,0:1]**3

def u_xx(x): #x = (x, t)
    return np.zeros((x.shape[0], 1)) +  6 * x[:,0:1] * x[:,1:2]**3

def f_func(x, alpha=-1.0, beta=0.0, gamma=1.0, k=3.0):
    return u_tt(x) + alpha * u_xx(x) + beta * u_func(x) + gamma * u_func(x)**k



def sampler(num_r, num_b, num_0, lx, rx, lt, rt, delta_N = 1001):
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
    UXlb = kg_equation(Xlb[:,0:1],Xlb[:,1:2])
    UXrb = kg_equation(Xrb[:,0:1],Xrb[:,1:2])
    
    xb = np.linspace(lx, rx, num_0)
    # T boundaries
    tlb   = lt*np.ones((xb.shape))
    Xic  = np.vstack((xb,tlb)).T
    Uic = kg_equation(Xic[:,0:1],Xic[:,1:2])

    # training tensors
    idxs = np.random.choice(xx.size, num_r, replace=False)
    X_train = torch.tensor(X[idxs], dtype=torch.float32, requires_grad=True,device=device)
    
    X_lb = torch.tensor(Xlb, dtype=torch.float32, device=device)
    X_rb = torch.tensor(Xrb, dtype=torch.float32, device=device)
    
    X_ic = torch.tensor(Xic, dtype=torch.float32, requires_grad=True,device=device)
    
    # compute mean and std of training data
    X_mean = torch.tensor(np.mean(np.concatenate([X[idxs], Xlb, Xrb, Xic], 0), axis=0, keepdims=True), dtype=torch.float32, device=device)
    
    X_std  = torch.tensor(np.std(np.concatenate([X[idxs], Xlb, Xrb, Xic], 0), axis=0, keepdims=True), dtype=torch.float32, device=device)
    
    U_Train= torch.tensor(f_func(X[idxs]), dtype=torch.float32, requires_grad=True,device=device)
    
    U_X_lb = torch.tensor(UXlb, dtype=torch.float32, device=device).reshape(num_b,1)
    U_X_rb = torch.tensor(UXrb, dtype=torch.float32, device=device).reshape(num_b,1)
    
    U_ic = torch.tensor(Uic, dtype=torch.float32, requires_grad=True, device=device).reshape(num_0,1)
    
    
    return X_train, X_lb, X_rb, X_ic, U_Train, U_X_lb, U_X_rb, U_ic, X_mean, X_std


###### computes pde residual
def KG_res(uhat, data): 
    x = data[:,0:1]
    t = data[:,1:2]
    
    poly = torch.ones_like(uhat)
    
    du = grad(outputs=uhat, inputs=data, 
              grad_outputs=torch.ones_like(uhat), create_graph=True)[0]
    
    dudx = du[:,0:1]
    dudt = du[:,1:2]
    
    dudxx = grad(outputs=dudx, inputs=data, 
              grad_outputs=torch.ones_like(uhat), create_graph=True)[0][:,0:1]
    dudtt = grad(outputs=dudt, inputs=data, 
              grad_outputs=torch.ones_like(uhat), create_graph=True)[0][:,1:2]
    
    residual = dudtt - dudxx + uhat**3  
                
    return residual


def KG_res_u_t(uhat, data): #data=(x,t)
    poly = torch.ones_like(uhat)
    
    du = grad(outputs=uhat, inputs=data, 
              grad_outputs=torch.ones_like(uhat), create_graph=True)[0]
    
    
    dudt = du[:,1:2]
    return dudt      




all_losses=[]
list_of_l2_Errors=[]

lr = 1e-3 
mm         = 100
i_print = 100

alpha_ann  = 0.5
n_epochs   = 20000 
num_r = 10000
num_b = 1000
num_0 = 1000
layer_sizes = [2,50,50,50,1] 


guding_lr = False
if guding_lr:
    path_loc= './results/guding_lr_%s_rb0_%s_%s_%s_iter_%s_%s' % (lr, num_r, num_b, num_0, n_epochs, layer_sizes) 
else:
    path_loc= './results/step_lr_%s_rb0_%s_%s_%s_iter_%s_%s' % (lr, num_r, num_b, num_0, n_epochs, layer_sizes) 


print('guding_lr, lr: ', guding_lr, lr)
print('num_r, num_b, num_0: ', num_r, num_b, num_0)
print('layer_sizes: ', layer_sizes)

if not os.path.exists(path_loc):
    os.makedirs(path_loc)


x = np.linspace(lx, rx, 1001) 
t = np.linspace(lt, rt, 1001) 
xx,tt = np.meshgrid(x,t)
u_sol = kg_equation(xx,tt)

X = np.vstack([xx.ravel(), tt.ravel()]).T


method_list = [0, 1, 2, 3, 'DB_PINN_mean', 'DB_PINN_std', 'DB_PINN_kurt']
#0: vanilla PINN (Equal Weighting); GW-PINN: 1: mean (max/avg); 2: std; 3: kurtosis;  

for i in range(7): 
    method = method_list[i]                       
    for j in range(1):
        
        print('i, j, method: ', i, j, method)
        save_loc = path_loc + '/method_' + str(method)  +  '/run_' + str(j) 
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)
        
        extras=str(num_r)+ "+"+ str(num_b) + "+" + str(num_0)
        print("#######Training with#####\n",extras)
        
        X_train, X_lb, X_rb, X_ic, U_Train,U_X_lb, U_X_rb, U_ic, X_mean, X_std= sampler(num_r, num_b, num_0, lx, rx, lt, rt)
        net = PINN(sizes=layer_sizes, mean=X_mean, std=X_std, activation=torch.nn.Tanh()).to(device)
        
        lambd_r        = torch.ones(1, device=device) 
        lambd_bc       = torch.ones(1, device=device)
        lambd_ic       = torch.ones(1, device=device)
        lambd_r_all       = [];
        lambd_bc_all      = [];
        lambd_ic_all      = [];
        
        losses = []
        losses_initial  = [];
        losses_boundary  = [];
        losses_residual = [];
        l2_error = []
        
        
        N_l = 0
        params = [{'params': net.parameters(), 'lr': lr}] 
        milestones = [[10000,20000,30000]]
        
        if guding_lr:
            optimizer = Adam(params)
        else:
            optimizer = Adam(params)
            scheduler = MultiStepLR(optimizer, milestones[0], gamma=0.1)
        
        print("training with shape of residual points: ", X_train.size())
        print("training with shape of boundary points (*2): ", X_lb.size())
        print("training with shape of initial points: ", X_ic.size())
        
        start_time = time.time()
        for epoch in range(n_epochs): 
            
            uhat  = net(X_train)
            res   = KG_res(uhat, X_train)
            l_reg = torch.mean((res-U_Train)**2)
            
            predl = net(X_lb)
            predr = net(X_rb)
            
            l_bc  = torch.mean((predl - U_X_lb)**2)  
            l_bc += torch.mean((predr - U_X_rb)**2)
            
            pred_ic = net(X_ic)
            l_ic = torch.mean((pred_ic - U_ic)**2) 
            
            gpreds= KG_res_u_t(pred_ic, X_ic)
            l_ic += torch.mean((gpreds)**2)
            
            L_t = torch.stack((l_reg, l_bc, l_ic))
            
            
            with torch.no_grad():
                if epoch % mm == 0:
                    
                    N_l += 1
                    stdr,kurtr=loss_grad_stats(l_reg, net)
                    stdb,kurtb=loss_grad_stats(l_bc, net)
                    stdi,kurti=loss_grad_stats(l_ic, net)
                    
                    maxr,meanr=loss_grad_max_mean(l_reg, net)
                    maxb,meanb=loss_grad_max_mean(l_bc, net,lambg=lambd_bc)
                    maxi,meani=loss_grad_max_mean(l_ic, net,lambg=lambd_ic)
                    
                    if epoch == 0:
                        lam_avg_bc = torch.zeros(1, device=device)
                        lam_avg_ic = torch.zeros(1, device=device)
                        running_mean_L = torch.zeros(1, device=device)
                        
                    if method == 1:
                        # max/avg
                        lamb_hat = maxr/meanb
                        lambd_bc     = (1-alpha_ann)*lambd_bc + alpha_ann*lamb_hat 
                        lamb_hat = maxr/meani
                        lambd_ic     = (1-alpha_ann)*lambd_ic + alpha_ann*lamb_hat 
                        
                    elif method == 2:
                        # inverse dirichlet
                        lamb_hat = stdr/stdb
                        lambd_bc     = (1-alpha_ann)*lambd_bc + alpha_ann*lamb_hat
                        lamb_hat = stdr/stdi
                        lambd_ic     = (1-alpha_ann)*lambd_ic + alpha_ann*lamb_hat
                    
                    elif method == 3:
                        # kurtosis based weighing
                        covr= stdr/kurtr
                        covb= stdb/kurtb
                        covi= stdi/kurti
                        lamb_hat = covr/covb
                        lambd_bc     = (1-alpha_ann)*lambd_bc + alpha_ann*lamb_hat
                        lamb_hat = covr/covi
                        lambd_ic     = (1-alpha_ann)*lambd_ic + alpha_ann*lamb_hat
                    
                    
                    elif method == 'DB_PINN_mean': 
                        
                        hat_all = maxr/meanb + maxr/meani
                        mean_param = (1. - 1 / N_l)
                        running_mean_L = mean_param * running_mean_L + (1 - mean_param) * L_t.detach()
                        l_t_vector = L_t/running_mean_L
                        hat_bc = hat_all* l_t_vector[1]/(l_t_vector[1] + l_t_vector[2])
                        hat_ic = hat_all* l_t_vector[2]/(l_t_vector[1] + l_t_vector[2])
                        lambd_bc = lam_avg_bc + 1/N_l*(hat_bc - lam_avg_bc)
                        lambd_ic = lam_avg_ic + 1/N_l*(hat_ic - lam_avg_ic)
                        lam_avg_bc = lambd_bc
                        lam_avg_ic = lambd_ic
                    
                    elif method == 'DB_PINN_std':  
                        
                        hat_all = stdr/stdb + stdr/stdi
                        mean_param = (1. - 1 / N_l)
                        running_mean_L = mean_param * running_mean_L + (1 - mean_param) * L_t.detach()
                        l_t_vector = L_t/running_mean_L
                        hat_bc = hat_all* l_t_vector[1]/(l_t_vector[1] + l_t_vector[2])
                        hat_ic = hat_all* l_t_vector[2]/(l_t_vector[1] + l_t_vector[2])
                        lambd_bc = lam_avg_bc + 1/N_l*(hat_bc - lam_avg_bc)
                        lambd_ic = lam_avg_ic + 1/N_l*(hat_ic - lam_avg_ic)
                        lam_avg_bc = lambd_bc
                        lam_avg_ic = lambd_ic
                    
                    
                    elif method == 'DB_PINN_kurt':  
                        
                        covr= stdr/kurtr
                        covb= stdb/kurtb
                        covi= stdi/kurti
                        hat_all = covr/covb + covr/covi
                        mean_param = (1. - 1 / N_l)
                        running_mean_L = mean_param * running_mean_L + (1 - mean_param) * L_t.detach()
                        l_t_vector = L_t/running_mean_L
                        hat_bc = hat_all* l_t_vector[1]/(l_t_vector[1] + l_t_vector[2])
                        hat_ic = hat_all* l_t_vector[2]/(l_t_vector[1] + l_t_vector[2])
                        lambd_bc = lam_avg_bc + 1/N_l*(hat_bc - lam_avg_bc)
                        lambd_ic = lam_avg_ic + 1/N_l*(hat_ic - lam_avg_ic)
                        lam_avg_bc = lambd_bc
                        lam_avg_ic = lambd_ic
            
                    else:
                        # equal weighting 
                        lambd_bc = torch.ones(1, device=device)
                        lambd_ic = torch.ones(1, device=device)
            
            
            loss = l_reg + lambd_bc.item()*l_bc + lambd_ic.item()*l_ic
            
            if epoch%i_print==0:
                
                inp = torch.tensor(X, dtype=torch.float32, device=device, requires_grad=True)
                out = net(inp).cpu().data.numpy().reshape(u_sol.shape)
                tmp = np.linalg.norm(out.reshape(-1)-u_sol.reshape(-1))/np.linalg.norm(out.reshape(-1))
                
                l2_error.append(tmp)
                list_of_l2_Errors.append(tmp)
                all_losses.append(loss.item())
                
                losses_initial.append(l_ic.item())
                losses_boundary.append(l_bc.item())
                losses_residual.append(l_reg.item())
                
                
                lambd_r_all.append(lambd_r.item())
                lambd_bc_all.append(lambd_bc.item())
                lambd_ic_all.append(lambd_ic.item())
                
                print("epoch {}/{}, loss={:.4f}, loss_r={:.6f}, loss_bc={:.6f}, loss_ic={:.6f}, lam_r={:.4f}, lam_bc={:.4f}, lam_ic={:.4f}, lr={:.5f}, l2_error(%)={:.3f}".format(epoch+1, n_epochs, loss.item(), l_reg.item(), l_bc.item(), l_ic.item(), lambd_r.item(), lambd_bc.item(), lambd_ic.item(), optimizer.param_groups[0]['lr'], tmp*100 ))
            
            
            optimizer.zero_grad()
            loss.backward()
            if guding_lr:
                optimizer.step()
            else:
                optimizer.step()
                scheduler.step()
                  
        elapsed_time = time.time() - start_time
        inp = torch.tensor(X, dtype=torch.float32, device=device, requires_grad=True)
        out = net(inp).cpu().data.numpy().reshape(u_sol.shape)
        
        print("\n.....\n")
        print("Method: , j: ",method, j)
        print("Relative L2 error = {:e}\n".format(np.linalg.norm(out.reshape(-1)-u_sol.reshape(-1))/np.linalg.norm(u_sol.reshape(-1))))
        print("Mean absolute error = {:e}\n".format(np.mean(np.abs(out.reshape(-1)-u_sol.reshape(-1)))))
        print("\n.....\n")
        
        
        torch.save(net.state_dict(), os.path.join(save_loc, 'model.pth'))
        
        
        U_star = u_sol.reshape(xx.shape)
        U_pred = out.reshape(xx.shape)
        
        ###########plot results
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
        ax.plot(losses_residual, label='$\mathcal{L}_{r}$')
        ax.plot(losses_boundary, label='$\mathcal{L}_{bc}$')
        ax.plot(losses_initial, label='$\mathcal{L}_{ic}$')
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
        
        
