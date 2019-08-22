from __future__ import print_function
import torch
import torch.nn as nn
import os
import sys
import t3nsor as t3
import torch.nn.functional as F

#partial TT net, input is in dense format, only weight is in TT format
class P_TT_Net(nn.Module):
    #init_flag=1: we will feed the NN a given initilization from outside
    def __init__(self,init1=None,init2=None,init3=None):
        super(P_TT_Net,self).__init__()
        self.ip1 = t3.TTLinear(in_features=28*28, out_features=300,init=init1)
        self.relu_ip1 = nn.ReLU(inplace=True)
        self.ip2 = t3.TTLinear(in_features=300,out_features=100,init=init2)
        self.relu_ip2 = nn.ReLU(inplace=True)
        self.ip3 = t3.TTLinear(in_features=100,out_features=10,init=init3)

    def forward(self,x):
        x = x.view(x.size(0), 28*28)
        x = self.ip1(x)
        x = self.relu_ip1(x)
        x = self.ip2(x)
        x = self.relu_ip2(x)
        x = self.ip3(x)
        return F.log_softmax(x,dim=1)

class Dense_Net(nn.Module):
    def __init__(self,init=None):
        super(Dense_Net, self).__init__()
        self.ip1 = nn.Linear(28*28, 300,init)
        self.relu_ip1 = nn.ReLU(inplace=True)
        self.ip2 = nn.Linear(300, 100,init)
        self.relu_ip2 = nn.ReLU(inplace=True)
        self.ip3 = nn.Linear(100, 10,init)
        return

    def forward(self, x):
        x = x.view(x.size(0), 28*28)
        x = self.ip1(x)
        x = self.relu_ip1(x)
        x = self.ip2(x)
        x = self.relu_ip2(x)
        x = self.ip3(x)
        return F.log_softmax(x,dim=1)


class OptNet(nn.Module):
    def __init__(self,init=None):
        super(OptNet,self).__init__()
        self.ip1 = t3.Solver(in_features=28*28,out_features=300)
        self.ip2 = t3.Solver(in_features=300,out_features=100)
        self.ip3 = t3.Solver(in_features=100,out_features=10)

    def forward(self,x):
        x = x.view(x.size(0), 28*28)
        x = self.ip1(x)
        x = self.ip2(x)
        x = self.ip3(x)
        return F.log_softmax(x,dim=1)
