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
lx=ly=-1
rx=ry=1

a_1 = 1
a_2 = 4
k = 1    
lam = k**2


# genereate ground truth
def generate_u(x):
    return np.sin(a_1 * np.pi * x[:, 0:1]) * np.sin(a_2 * np.pi * x[:, 1:2])

def u_xx(x, a_1, a_2):
    return - (a_1 * np.pi) ** 2 * np.sin(a_1 * np.pi * x[:, 0:1]) * np.sin(a_2 * np.pi * x[:, 1:2])

def u_yy(x, a_1, a_2):
    return - (a_2 * np.pi) ** 2 * np.sin(a_1 * np.pi * x[:, 0:1]) * np.sin(a_2 * np.pi * x[:, 1:2])

# Forcing
def f(x, a_1, a_2, lam):
    return u_xx(x, a_1, a_2) + u_yy(x, a_1, a_2) + lam * generate_u(x, a_1, a_2)




x = np.linspace(lx, rx, 1001)[:, None] 
y = np.linspace(ly, ry, 1001)[:, None] 
xx,yy = np.meshgrid(x,y)
X = np.hstack((xx.flatten()[:, None], yy.flatten()[:, None]))
u_sol = generate_u(X)





def sampler(num_r, num_b, u_func, lx ,rx, ly, ry, delta_N = 1001): 
    # generate training data
    x = np.linspace(lx, rx, delta_N)
    y = np.linspace(ly, ry, delta_N)
    xb = np.linspace(lx,rx,num_b) 
    yb = np.linspace(ly,ry,num_b)
    
    xx,yy = np.meshgrid(x,y)
    
    X = np.vstack([xx.ravel(), yy.ravel()]).T
    
    # X boundaries
    lb   = lx*np.ones((yb.shape))
    rb   = rx*np.ones((yb.shape))
    Xlb  = np.vstack((lb,yb)).T
    Xrb  = np.vstack((rb,yb)).T
    UXlb = u_func(Xlb)
    UXrb = u_func(Xrb)
    
    # Y boundaries
    lb   = ly*np.ones((xb.shape))
    rb   = ry*np.ones((xb.shape))
    Ylb  = np.vstack((xb,lb)).T
    Yrb  = np.vstack((xb,rb)).T
    UYlb = u_func(Ylb)   
    UYrb = u_func(Yrb) 
    
    # training tensors
    idxs = np.random.choice(xx.size, num_r, replace=False)
    X_train = torch.tensor(X[idxs], dtype=torch.float32, requires_grad=True,device=device)
    X_rb = torch.tensor(Xrb, dtype=torch.float32, device=device)
    X_lb = torch.tensor(Xlb, dtype=torch.float32, device=device)
    Y_rb = torch.tensor(Yrb, dtype=torch.float32, device=device)
    Y_lb = torch.tensor(Ylb, dtype=torch.float32, device=device)
    # compute mean and std of training data
    X_mean = torch.tensor(np.mean(np.concatenate([X[idxs], Xrb, Xlb, Yrb, Ylb], 0), axis=0, keepdims=True), dtype=torch.float32, device=device)
    X_std  = torch.tensor(np.std(np.concatenate([X[idxs], Xrb, Xlb, Yrb, Ylb], 0), axis=0, keepdims=True), dtype=torch.float32, device=device)
    
    U_X_rb = torch.tensor(UXrb, dtype=torch.float32, device=device).reshape(num_b,1)
    U_X_lb = torch.tensor(UXlb, dtype=torch.float32, device=device).reshape(num_b,1)
    U_Y_rb = torch.tensor(UYrb, dtype=torch.float32, device=device).reshape(num_b,1)
    U_Y_lb = torch.tensor(UYlb, dtype=torch.float32, device=device).reshape(num_b,1)
    
    
    return X_train, X_lb, X_rb, Y_lb, Y_rb, U_X_lb, U_X_rb, U_Y_lb, U_Y_rb, X_mean, X_std




# computes pde residual
def Helmholtz_res(uhat, data):
    x = data[:,0:1]
    y = data[:,1:2]
    du = grad(outputs=uhat, inputs=data, grad_outputs=torch.ones_like(uhat), create_graph=True)[0]
    dudx = du[:,0:1]
    dudxx = grad(outputs=dudx, inputs=data,grad_outputs=torch.ones_like(uhat), create_graph=True)[0][:,0:1]
    dudy = du[:,1:2]
    dudyy = grad(outputs=dudy, inputs=data,grad_outputs=torch.ones_like(uhat), create_graph=True)[0][:,1:2]
    source = - (a_1*math.pi)**2*torch.sin(a_1*math.pi*x)*torch.sin(a_2*math.pi*y) - \
                (a_2*math.pi)**2*torch.sin(a_1*math.pi*x)*torch.sin(a_2*math.pi*y) + \
                lam*torch.sin(a_1*math.pi*x)*torch.sin(a_2*math.pi*y)
    residual = dudxx + dudyy  + lam*uhat - source
    return residual


lr = 1e-3
mm         = 10  
alpha_ann  = 0.5
Adam_n_epochs   = 30000 
i_print = 100

N_r = 20000
N_bc = 100
layer_sizes = [2,50,50,50,50,1]  

guding_lr = False
lr_gamma = 0.1

if guding_lr:
    path_loc= './results/guding_lr_%s_A-%s' % (lr, Adam_n_epochs) 
else:
    path_loc= './results/step_lr_%s_gamma_%s_A-%s' % (lr, lr_gamma, Adam_n_epochs) 

print('guding_lr, lr: ', guding_lr, lr)


method_list = [0, 1, 2, 3] 
#0: vanilla PINN (Equal Weighting); GW-PINN: 1: mean (max/avg); 2: std; 3: kurtosis; 

for i in range(4):
    method = method_list[i]
    for j in range(1):
        print('i,j, method: ', i, j, method)
        save_loc = path_loc + '/method_' + str(method) + '/run_' + str(j) 
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)
        
        
        X_train, X_lb, X_rb, Y_lb, Y_rb, U_X_lb, U_X_rb, U_Y_lb, U_Y_rb, X_mean, X_std= sampler(num_r=N_r, num_b=N_bc, u_func = generate_u, lx=lx, rx=rx,ly=ly,ry=ry)
        net = PINN(sizes=layer_sizes, mean=X_mean, std=X_std, activation=torch.nn.Tanh()).to(device) 
        lambd       = torch.ones(1, device=device)  
        lambds      = []
        losses = []
        losses_boundary  = []
        losses_residual = []
        l2_error = []
        
        params = [{'params': net.parameters(), 'lr': lr}]
        milestones = [[10000,20000,30000]]

        if guding_lr:
            optimizer = Adam(params) 
        else:
            optimizer = Adam(params) 
            scheduler = MultiStepLR(optimizer, milestones[0], gamma=lr_gamma)
            
        print("training with shape of residual points", X_train.size())
        print("training with shape of boundary points (*4)", X_lb.size())
        start_time = time.time()
        for epoch in range(Adam_n_epochs):     
            uhat  = net(X_train)
            res   = Helmholtz_res(uhat, X_train)
            l_reg = torch.mean((res)**2)
            predl = net(X_lb)
            predr = net(X_rb)
            l_bc  = torch.mean((predl - U_X_lb)**2)
            l_bc += torch.mean((predr - U_X_rb)**2)
            predl = net(Y_lb)
            predr = net(Y_rb)
            l_bc += torch.mean((predl - U_Y_lb)**2)
            l_bc += torch.mean((predr - U_Y_rb)**2)  

            
            with torch.no_grad():
                if epoch % mm == 0:
                    stdr,kurtr=loss_grad_stats(l_reg, net)
                    stdb,kurtb=loss_grad_stats(l_bc, net)
                    maxr,meanr=loss_grad_max_mean(l_reg, net)
                    maxb,meanb=loss_grad_max_mean(l_bc, net,lambg=lambd)
                    
                    if method == 1:
                        # max/avg
                        lamb_hat = maxr/meanb
                        lambd     = (1-alpha_ann)*lambd + alpha_ann*lamb_hat 
                        
                    elif method == 2:
                        # inverse dirichlet
                        lamb_hat = stdr/stdb
                        lambd     = (1-alpha_ann)*lambd + alpha_ann*lamb_hat
                            
                    elif method == 3:
                        # kurtosis based weighing
                        covr= stdr/kurtr
                        covb= stdb/kurtb
                        lamb_hat = covr/covb
                        lambd     = (1-alpha_ann)*lambd + alpha_ann*lamb_hat
                            
                    else:
                        # equal weighting 
                        lambd = torch.ones(1, device=device)
            
            if lambd < 1 or torch.isnan(lambd) or torch.isinf(lambd):
                lambd = torch.tensor(1.0, dtype=torch.float32, device=device)
            
            loss = l_reg + lambd*l_bc
            
            
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
                losses_boundary.append(l_bc.item())
                losses_residual.append(l_reg.item())
                
                lambds.append(lambd.item())
                losses.append(loss.item())
                
                print("Adam optimizing:   epoch {}/{}, loss={:.6f}, loss_bc={:.6f},loss_r={:.6f}, lambda={:.4f}, lr={:,.7f}, L2 error (%)={:.6f}".format(epoch+1, Adam_n_epochs, loss.item(), l_bc.item(), l_reg.item(), lambd.item(), optimizer.param_groups[0]['lr'], tmp*100)) 
                
                
        elapsed_time = time.time() - start_time
        print('Adam training time = ',elapsed_time)
        
        inp = torch.tensor(X, dtype=torch.float32, device=device)
        out = net(inp).cpu().data.numpy().reshape(u_sol.shape)
        print("\n...Adam training...\n")
        print("Method: , j: ",method, j)
        print("pred rel. l2-error = {:e}\n".format(np.linalg.norm(out.reshape(-1)-u_sol.reshape(-1))/np.linalg.norm(u_sol.reshape(-1))))
        print("pred abs. error = {:e}\n".format(np.mean(np.abs(out.reshape(-1)-u_sol.reshape(-1)))))
        print("\n.....\n")
        
        
        U_star = u_sol.reshape(xx.shape)
        U_pred = out.reshape(xx.shape)
        
        ###########
        fig = plt.figure(1, figsize=(18, 5))
        fig_1 = plt.subplot(1, 3, 1)
        plt.pcolor(xx, yy, U_star, cmap='jet')
        plt.colorbar()
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.title('Exact $u(x)$')
        fig_2 = plt.subplot(1, 3, 2)
        plt.pcolor(xx, yy, U_pred, cmap='jet')
        plt.colorbar()
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.title('Predicted $u(x)$')
        fig_3 = plt.subplot(1, 3, 3)
        plt.pcolor(xx, yy, np.abs(U_star - U_pred), cmap='jet')
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
        ax.plot(losses_boundary, label='$\mathcal{L}_{u_b}$')
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
        ax.plot(lambds, label='$\lambda_{u_b}$')
        ax.set_xlabel('iterations')
        plt.tight_layout()
        plt.savefig(os.path.join(save_loc,'3.bc_weights.png'))
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
        

