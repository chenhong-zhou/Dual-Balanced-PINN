import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
import math


# sin activation
class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)


class Layer(nn.Module):
    def __init__(self, in_features, out_features, activation):
        super(Layer, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        gain = 5/3 if isinstance(activation, nn.Tanh) else 1
        nn.init.xavier_normal_(self.linear.weight, gain=gain)
        nn.init.zeros_(self.linear.bias)
        self.linear = self.linear
        
    def forward(self, x):
        return self.linear(x)



class PINN(nn.Module):
    def __init__(self, sizes, mean=None, std=None, activation=nn.Tanh()):
        super(PINN, self).__init__()
        
        self.mean=mean  
        self.std= std  
        
        if activation == 'sin':
            activation = Sine()
        elif activation == 'relu':
            activation = torch.nn.ReLU()
        
        layer = []
        for i in range(len(sizes)-2):
            linear = Layer(sizes[i], sizes[i+1], activation)
            layer += [linear, activation]
        layer += [Layer(sizes[-2], sizes[-1], activation)]
        self.net = nn.Sequential(*layer)
        
        
    def forward(self, x):
        if self.mean is None:
            X = x
        else:
            X = (x-self.mean)/self.std
        
        return self.net(X)
    
