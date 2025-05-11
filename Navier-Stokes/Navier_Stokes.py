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
from torch.optim import Adam, LBFGS
from torch.optim.lr_scheduler import StepLR, ExponentialLR, MultiStepLR

from scipy.interpolate import griddata

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# experiment setup
lx=ly=0.0
rx=ry=1.0



u_ref= np.genfromtxt("../data/Lid-driven-Cavity/reference_u.csv", delimiter=',')    
v_ref= np.genfromtxt("../data/Lid-driven-Cavity/reference_v.csv", delimiter=',')   
velocity_ref = np.sqrt(u_ref**2 + v_ref**2)  


nx = 100
ny = 100 
x = np.linspace(lx, rx, nx)
y = np.linspace(ly, ry, ny)
X, Y = np.meshgrid(x, y)
X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))
velocity_sol = velocity_ref.T.flatten()[:,None]
u_sol = u_ref.T.flatten()[:,None]
v_sol = v_ref.T.flatten()[:,None]

Re = 100.0




def sampler(num_r, num_b):   
    # generate training data
    xb = np.linspace(lx,rx,num_b)
    yb = np.linspace(ly,ry,num_b)
    
    # X boundaries
    lb   = lx*np.ones((yb.shape))
    rb   = rx*np.ones((yb.shape))
    Xlb  = np.vstack((lb,y)).T
    Xrb  = np.vstack((rb,y)).T
    UXlb = np.zeros((Xlb.shape[0], 2))
    UXrb = np.zeros((Xrb.shape[0], 2))
    
    # Y boundaries
    lb   = ly*np.ones((xb.shape))
    rb   = ry*np.ones((xb.shape))
    Ylb  = np.vstack((xb,lb)).T
    Yrb  = np.vstack((xb,rb)).T   
    UYlb = np.zeros((Ylb.shape[0], 2)) 
    UYrb = np.tile(np.array([1.0, 0.0]), (Yrb.shape[0], 1))
    
    # training tensors
    from pyDOE import lhs
    lb = np.array([lx, ly])
    ub = np.array([rx, ry])
    X_f_np = lb + (ub-lb)*lhs(2, num_r)
    X_f = torch.tensor(X_f_np, dtype=torch.float32, requires_grad=True, device=device)
    
    X_rb = torch.tensor(Xrb, dtype=torch.float32, requires_grad=True, device=device)
    X_lb = torch.tensor(Xlb, dtype=torch.float32, requires_grad=True, device=device)
    Y_rb = torch.tensor(Yrb, dtype=torch.float32, requires_grad=True, device=device)
    Y_lb = torch.tensor(Ylb, dtype=torch.float32, requires_grad=True, device=device)


    # compute mean and std of training data
    X_mean = torch.tensor(np.mean(np.concatenate([X_f_np, Xrb, Xlb, Yrb, Ylb], 0), axis=0, keepdims=True), dtype=torch.float32, device=device)
    X_std  = torch.tensor(np.std(np.concatenate([X_f_np, Xrb, Xlb, Yrb, Ylb], 0), axis=0, keepdims=True), dtype=torch.float32, device=device)
    
    U_Y_rb = torch.tensor(UYrb, dtype=torch.float32, device=device).reshape(num_b, 2)
    
    return X_f, X_lb, X_rb, Y_lb, Y_rb, U_Y_rb, X_mean, X_std  




# computes pde residual
def NS_res(uhat, data): #data: (x,y)   uhat: (psi, p)
    x = data[:,0:1]
    y = data[:,1:2]
    psi = uhat[:,0:1]
    p = uhat[:,1:2]
    
    psi_xy = grad(outputs=psi, inputs=data, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
    u = psi_xy[:, 1:2]  #psi_y
    v = -psi_xy[:, 0:1] #-psi_x
    
    u_x = grad(outputs=u, inputs=data, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 0:1] 
    u_y = grad(outputs=u, inputs=data, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 1:2] 
    
    v_x = grad(outputs=v, inputs=data, grad_outputs=torch.ones_like(v), create_graph=True)[0][:, 0:1] 
    v_y = grad(outputs=v, inputs=data, grad_outputs=torch.ones_like(v), create_graph=True)[0][:, 1:2]  
    
    p_x = grad(outputs=p, inputs=data, grad_outputs=torch.ones_like(p), create_graph=True)[0][:, 0:1]  
    p_y = grad(outputs=p, inputs=data, grad_outputs=torch.ones_like(p), create_graph=True)[0][:, 1:2]  

    u_xx = grad(outputs=u_x, inputs=data, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1] 
    u_yy = grad(outputs=u_y, inputs=data, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1:2]   

    v_xx = grad(outputs=v_x, inputs=data, grad_outputs=torch.ones_like(v_x), create_graph=True)[0][:, 0:1] 
    v_yy = grad(outputs=v_y, inputs=data, grad_outputs=torch.ones_like(v_y), create_graph=True)[0][:, 1:2] 
    
    u_res = u * u_x + v * u_y + p_x - (u_xx + u_yy) / Re
    v_res = u * v_x + v * v_y + p_y - (v_xx + v_yy) / Re
    return u_res, v_res


# computes pde dx dy
def NS_uv(uhat, data):  #data: (x,y)   uhat: (psi, p)
    x = data[:,0:1]
    y = data[:,1:2]
    psi = uhat[:,0:1]
    p = uhat[:,1:2]
    
    psi_xy = grad(outputs=psi, inputs=data, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
    u = psi_xy[:, 1:2]  #psi_y
    v = -psi_xy[:, 0:1] #-psi_x
    return u, v



lr = 1e-3
mm         = 10   
i_print = 100
alpha_ann  = 0.1
Adam_n_epochs   = 40000 

N_r = 20000
N_bc = 100
layer_sizes = [2, 50, 50, 50, 2] 

lr_lbfgs = 0.1


guding_lr = True
lr_gamma = 0.1

if guding_lr:
    path_loc= './results/rb_%s_%s_%s/guding_lr_%s_A-%s' % (N_r, N_bc, layer_sizes, lr, Adam_n_epochs) 
else:
    path_loc= './results/rb_%s_%s_%s/step_lr_%s_gamma_%s_A-%s' % (N_r, N_bc, layer_sizes, lr,lr_gamma, Adam_n_epochs) 

print('guding_lr, lr: ', guding_lr, lr)

if not os.path.exists(path_loc):
    os.makedirs(path_loc)



method_list = ['DB_PINN_mean', 'DB_PINN_std', 'DB_PINN_kurt'] 

for i in range(3): 
    method = method_list[i]
    for j in range(1):
        
        print('i,j, method: ', i, j, method)
        save_loc = path_loc + '/method_' + str(method) + '/run_' + str(j) 
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)
        
        X_train, X_lb, X_rb, Y_lb, Y_rb, U_Y_rb, X_mean, X_std= sampler(num_r=N_r, num_b=N_bc)
        net = PINN(sizes=layer_sizes, mean=X_mean, std=X_std, activation=torch.nn.Tanh()).to(device) 
        lambd_bc_0 = torch.ones(1, device=device)  
        lambd_bc_1 = torch.ones(1, device=device) 
        
        lambds_bc_0      = []
        lambds_bc_1      = []
        losses = []

        losses_residual = []
        losses_boundary_0  = []
        losses_boundary_1  = []
        
        l2_error_velocity = []
        l2_error_u = []
        l2_error_v = []
        
        params = [{'params': net.parameters(), 'lr': lr}]
        milestones = [[10000,20000,30000]]

        if guding_lr:
            optimizer = Adam(params) 
        else:
            optimizer = Adam(params) 
            scheduler = MultiStepLR(optimizer, milestones[0], gamma=lr_gamma)
            
        N_l = 0
        
        print("training with shape of residual points", X_train.size())
        print("training with shape of boundary points (*4)", X_lb.size())
        
        start_time = time.time()
        for epoch in range(Adam_n_epochs):     
            uhat  = net(X_train)   #uhat: (psi, p)
            u_res, v_res   = NS_res(uhat, X_train) 
            l_reg = torch.mean((u_res)**2)  + torch.mean((v_res)**2) 
            
            predl = net(X_lb)
            pred_1, pred_2 = NS_uv(predl, X_lb)
            xlb_pred = torch.cat((pred_1, pred_2), dim=1)
            
            predr = net(X_rb)
            pred_1, pred_2 = NS_uv(predr, X_rb)
            xrb_pred = torch.cat((pred_1, pred_2), dim=1)
            
            l_bc_0  = torch.mean((xlb_pred)**2)
            l_bc_0 += torch.mean((xrb_pred)**2)
            
            predl = net(Y_lb)
            pred_1, pred_2 = NS_uv(predl, Y_lb)
            ylb_pred = torch.cat((pred_1, pred_2), dim=1)
            
            predr = net(Y_rb)
            pred_1, pred_2 = NS_uv(predr, Y_rb)
            yrb_pred = torch.cat((pred_1, pred_2), dim=1)
            
            l_bc_0 += torch.mean((ylb_pred)**2)
            l_bc_1 = torch.mean((yrb_pred - U_Y_rb)**2)

            L_t = torch.stack((l_reg, l_bc_0, l_bc_1))
            
            
            with torch.no_grad():
                if epoch % mm == 0:
                    N_l += 1
                    
                    stdr,kurtr=loss_grad_stats(l_reg, net)
                    stdb0,kurtb0=loss_grad_stats(l_bc_0, net)
                    stdb1,kurtb1=loss_grad_stats(l_bc_1, net)
                    
                    maxr,meanr=loss_grad_max_mean(l_reg, net)
                    maxb0,meanb0=loss_grad_max_mean(l_bc_0, net,lambg=lambd_bc_0)
                    maxb1,meanb1=loss_grad_max_mean(l_bc_1, net,lambg=lambd_bc_1)

                    if epoch == 0:
                        lam_avg_bc_0 = torch.zeros(1, device=device)
                        lam_avg_bc_1 = torch.zeros(1, device=device)
                        running_mean_L = torch.zeros(1, device=device)
                    
                    
                    if  method == 'DB_PINN_mean':
                        # max/avg
                        hat_all = maxr/meanb0 + maxr/meanb1
                        
                        mean_param = (1. - 1 /N_l)    
                        running_mean_L = mean_param * running_mean_L + (1 - mean_param) * L_t.detach()
                        l_t_vector = L_t/running_mean_L
                        hat_bc_0 = hat_all* l_t_vector[1]/torch.sum(l_t_vector[1:])
                        hat_bc_1 = hat_all* l_t_vector[2]/torch.sum(l_t_vector[1:])
                        
                        lambd_bc_0 = lam_avg_bc_0 + 1/N_l*(hat_bc_0 - lam_avg_bc_0)
                        lambd_bc_1 = lam_avg_bc_1 + 1/N_l*(hat_bc_1 - lam_avg_bc_1)
                        lam_avg_bc_0 = lambd_bc_0
                        lam_avg_bc_1 = lambd_bc_1
                        
                        
                    elif method == 'DB_PINN_std':
                        # max/avg
                        hat_all = stdr/stdb0 + stdr/stdb1
                        
                        mean_param = (1. - 1 /N_l)    
                        running_mean_L = mean_param * running_mean_L + (1 - mean_param) * L_t.detach()
                        l_t_vector = L_t/running_mean_L
                        hat_bc_0 = hat_all* l_t_vector[1]/torch.sum(l_t_vector[1:])
                        hat_bc_1 = hat_all* l_t_vector[2]/torch.sum(l_t_vector[1:])
                        
                        lambd_bc_0 = lam_avg_bc_0 + 1/N_l*(hat_bc_0 - lam_avg_bc_0)
                        lambd_bc_1 = lam_avg_bc_1 + 1/N_l*(hat_bc_1 - lam_avg_bc_1)
                        lam_avg_bc_0 = lambd_bc_0
                        lam_avg_bc_1 = lambd_bc_1
                        
                        
                    elif method == 'DB_PINN_kurt':
                        covr= stdr/kurtr
                        covb0= stdb0/kurtb0
                        covb1= stdb1/kurtb1
                        
                        hat_all = covr/covb0 + covr/covb1
                        mean_param = (1. - 1 /N_l)    
                        running_mean_L = mean_param * running_mean_L + (1 - mean_param) * L_t.detach()
                        l_t_vector = L_t/running_mean_L
                        hat_bc_0 = hat_all* l_t_vector[1]/torch.sum(l_t_vector[1:])
                        hat_bc_1 = hat_all* l_t_vector[2]/torch.sum(l_t_vector[1:])
                        
                        lambd_bc_0 = lam_avg_bc_0 + 1/N_l*(hat_bc_0 - lam_avg_bc_0)
                        lambd_bc_1 = lam_avg_bc_1 + 1/N_l*(hat_bc_1 - lam_avg_bc_1)
                        lam_avg_bc_0 = lambd_bc_0
                        lam_avg_bc_1 = lambd_bc_1
            
            loss = l_reg + lambd_bc_0*l_bc_0 + lambd_bc_1*l_bc_1
            
            
            optimizer.zero_grad()
            loss.backward()
            if guding_lr:
                optimizer.step()
            else:
                optimizer.step()
                scheduler.step()
                
            if epoch%i_print==0:
                
                inp = torch.tensor(X_star, dtype=torch.float32, requires_grad=True, device=device)
                out = net(inp)
                u_pred, v_pred = NS_uv(out, inp)
                velocity_pred = torch.sqrt(u_pred**2 + v_pred**2)
                
                tmp = np.linalg.norm(velocity_pred.cpu().data.numpy() - velocity_sol, 2)/np.linalg.norm(velocity_sol, 2)

                l2_error_velocity.append(tmp)
                
                losses_residual.append(l_reg.item())
                losses_boundary_0.append(l_bc_0.item())
                losses_boundary_1.append(l_bc_1.item())
                
                
                losses.append(loss.item())
                lambds_bc_0.append(lambd_bc_0.item())
                lambds_bc_1.append(lambd_bc_1.item())
                
                print("Adam optimizing: method={}, j={}, epoch {}/{}, loss={:.6f},loss_r={:.6f}, loss_bc_0={:.6f}, loss_bc_1={:.6f}, lam_bc_0={:.4f},lam_bc_1={:.4f}, lr={:,.7f}, L2 error (%)={:.6f}".format(method, j, epoch+1, Adam_n_epochs, loss.item(), l_reg.item(), l_bc_0.item(), l_bc_1.item(), lambd_bc_0.item(), lambd_bc_1.item(),  optimizer.param_groups[0]['lr'], tmp*100)) 
                
                
                
        elapsed_time = time.time() - start_time
        print('Adam training time = ',elapsed_time)

        inp = torch.tensor(X_star, dtype=torch.float32, requires_grad=True, device=device)
        out = net(inp)
        u_pred, v_pred = NS_uv(out, inp)
        velocity_pred = torch.sqrt(u_pred**2 + v_pred**2)
        
        l2_vel = np.linalg.norm(velocity_pred.cpu().data.numpy() - velocity_sol, 2)/np.linalg.norm(velocity_sol, 2)
        abs_vel = np.mean(np.abs(velocity_pred.cpu().data.numpy() - velocity_sol))
        
        print("\n...Adam training...\n")
        print("Method: , j: ",method, j)
        print("pred velocity rel. l2-error = {:e}\n".format(l2_vel))
        print("pred velocity abs. error = {:e}\n".format(abs_vel))
        print("\n.....\n")

        
        U_star = velocity_sol.reshape(X.shape)
        U_pred = velocity_pred.cpu().data.numpy().reshape(X.shape)
        
        ###########
        fig = plt.figure(1, figsize=(18, 5))
        fig_1 = plt.subplot(1, 3, 1)
        plt.pcolor(X, Y, U_star, cmap='jet')
        plt.colorbar()
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.title('Exact $u(x)$')
        fig_2 = plt.subplot(1, 3, 2)
        plt.pcolor(X, Y, U_pred, cmap='jet')
        plt.colorbar()
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.title('Predicted $u(x)$')
        fig_3 = plt.subplot(1, 3, 3)
        plt.pcolor(X, Y, np.abs(U_star - U_pred), cmap='jet')
        plt.colorbar()
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.title('Absolute error')
        plt.tight_layout()
        plt.savefig(os.path.join(save_loc,'1.predictions.png'))
        plt.show()
        plt.close()
        
        fig_2 = plt.figure(2)
        ax = fig_2.add_subplot(1, 1, 1)
        ax.plot(losses_residual, label='$\mathcal{L}_{r}$')
        ax.plot(losses_boundary_0, label='$\mathcal{L}_{u_b0}$')
        ax.plot(losses_boundary_1, label='$\mathcal{L}_{u_b1}$')
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
        ax.plot(lambds_bc_0, label='$\lambda_{u_b0}$')
        ax.plot(lambds_bc_1, label='$\lambda_{u_b1}$')
        ax.set_xlabel('iterations')
        plt.tight_layout()
        plt.savefig(os.path.join(save_loc,'3.bc_weights.png'))
        plt.show()
        plt.close()

        fig_4 = plt.figure(4)
        ax = fig_4.add_subplot(1, 1, 1)
        ax.plot(l2_error_velocity)
        ax.set_xlabel('iterations')
        plt.tight_layout()
        plt.savefig(os.path.join(save_loc,'4.l2_error_velocity.png'))
        plt.show()
        plt.close()
        
