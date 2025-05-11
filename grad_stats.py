import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad

def loss_grad_stats(loss, net):
    """
    Functionality: provides std, kurtosis of backpropagated gradients of loss function 
    inputs: loss: loss function ; net: the NN model
    outputs: std and kurtosis 
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grad_ = torch.zeros((0), dtype=torch.float64,device=device)
    
    for m in net.modules():
        if not isinstance(m, nn.Linear):
            continue
        if(m == 0):
            w = grad(loss, m.weight, retain_graph=True,allow_unused=True)[0]
            b = grad(loss, m.bias, retain_graph=True,allow_unused=True)[0]        
            grad_ = torch.cat((w.view(-1), b))
            
        else:
            w = grad(loss, m.weight, retain_graph=True,allow_unused=True)[0]
            b = grad(loss, m.bias, retain_graph=True,allow_unused=True)[0]  
            if w is not None and b is not None:   
                grad_ = torch.cat((grad_,w.view(-1), b))
            elif w is not None and b is None:  
                grad_ = torch.cat((grad_,w.view(-1)))
            elif w is None and b is not None: 
                grad_ = torch.cat((grad_, b))
    
    #collect gradient statistics
    mean = torch.mean(grad_)
    diffs = grad_ - mean
    std = torch.std(grad_)
    
    zscores = diffs / std
    kurtoses = torch.mean(torch.pow(zscores, 4.0))
            
    return std, kurtoses






def loss_grad_max_mean(loss, net, lambg=1):
    """
    Functionality: provides maximum and mean of backpropagated gradients of loss function 
    inputs: loss: loss function ; net: the NN model; lambg : term for weighted stats (optional)
    outputs: max and mean
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grad_ = torch.zeros((0), dtype=torch.float64,device=device)
    
    for m in net.modules():
        if not isinstance(m, nn.Linear):
            continue
        if(m == 0):
            w = torch.abs(lambg*grad(loss, m.weight, retain_graph=True, allow_unused=True)[0])
            b = torch.abs(lambg*grad(loss, m.bias, retain_graph=True, allow_unused=True)[0])        
            grad_ = torch.cat((w.view(-1), b))
        else:
            w = torch.abs(lambg*grad(loss, m.weight, retain_graph=True, allow_unused=True)[0])
            
            if grad(loss, m.bias, retain_graph=True, allow_unused=True)[0] is None:
                b = None
            else:
                b = torch.abs(lambg*grad(loss, m.bias, retain_graph=True, allow_unused=True)[0])    
            
            if b is not None:
                grad_ = torch.cat((grad_,w.view(-1), b))
            else:
                grad_ = torch.cat((grad_,w.view(-1)))
    
    maxgrad = torch.max(grad_)
    meangrad = torch.mean(grad_)
    return maxgrad,meangrad 