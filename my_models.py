from __future__ import print_function
import torch
import torch.nn as nn
import os
import sys
import t3nsor as t3
import torch.nn.functional as F

class pre_mani_net(nn.Module):
    def __init__(self,settings):
        super(pre_mani_net,self).__init__()
        self.to_tt = t3.layer_to_tt_tensor(settings)
        self.conv1 = t3.TTConv(settings.TT_SHAPE,in_channels=1,out_channels=16)
        # self.conv1 = nn.Conv2d(3,6,5)
        # self.pool = nn.MaxPool2d(2,2)
        # self.conv2 = nn.Conv2d(6,16,5)
        # self.conv3 = nn.Conv2d(16,3,5)
        self.fc1 = t3.TTFC(settings.TT_SHAPE,tt_rank=settings.TT_RANK,in_channels=16)
        #self.solver1 = t3.TTSolver(in_features=16,out_features=120)
        #self.solver2= t3.TTSolver(in_features=120,out_features=84)
        self.fc2 = t3.TTLinear(in_features=16,out_features=16)
        self.relu_ip1 = nn.ReLU(inplace=True)
        self.fc3 = t3.TTLinear(in_features=16,out_features=84)
        self.relu_ip2 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(84,10)

    def forward(self, x):
        x = self.to_tt(x)
        x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        #x = self.conv4(x)
        #x = self.conv5(x)
        x = self.fc1(x)
        #x = self.solver1(x)
        #x = self.solver2(x)
        x = self.fc2(x)
        x = self.relu_ip1(x)
        x = self.fc3(x)
        x = self.relu_ip2(x)
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

#add batchnorm
class manifold_Net(nn.Module):
    def __init__(self,settings):
        super(manifold_Net, self).__init__()
        #TODO:figure out exact arg value
        self.conv1 = t3.TTConv(settings.TT_SHAPE,in_channels=3,out_channels=16)
        # self.conv1 = nn.Conv2d(3,6,5)
        # self.pool = nn.MaxPool2d(2,2)
        # self.conv2 = nn.Conv2d(6,16,5)
        # self.conv3 = nn.Conv2d(16,3,5)
        self.fc1 = t3.TTFC(settings.TT_SHAPE,tt_rank=settings.TT_RANK,in_channels=10)
        #self.solver1 = t3.TTSolver(in_features=16,out_features=120)
        #self.solver2= t3.TTSolver(in_features=120,out_features=84)
        #self.fc2 = t3.TTLinear(in_features=84,out_features=16)
        #self.relu_ip1 = nn.ReLU(inplace=True)
        #self.fc3 = t3.TTLinear(in_features=16,out_features=84)
        #self.relu_ip2 = nn.ReLU(inplace=True)
        #self.fc4 = nn.Linear(84,10)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        #x = self.conv4(x)
        #x = self.conv5(x)
        x = self.fc1(x)
        #x = self.solver1(x)
        #x = self.solver2(x)
        #x = self.fc2(x)
        #x = self.relu_ip1(x)
        #x = self.fc3(x)
        #x = self.relu_ip2(x)
        #x = self.fc4(x)
        return F.log_softmax(x, dim=1)


class pre_FTT_Net(nn.Module):
    def __init__(self,settings):
        super(pre_FTT_Net,self).__init__()
        self.to_tt_matrix = t3.layer_to_tt_matrix(settings)
        self.ip1 = t3.FTTLinear(in_features=512, out_features=10)
        self.to_dense = t3.layers.tt_to_dense()
    def forward(self,x):
        x = self.to_tt_matrix(x)
        x = self.ip1(x)
        x = self.to_dense(x)
        return F.log_softmax(x,dim=1)


#partial TT net, input is in dense format, only weight is in TT format
class P_TT_Net(nn.Module):
    #init_flag=1: we will feed the NN a given initilization from outside
    def __init__(self):
        super(P_TT_Net,self).__init__()
        self.ip1 = t3.TTLinear(in_features=28*28, out_features=300)
        self.relu_ip1 = nn.ReLU(inplace=True)
        self.ip2 = t3.TTLinear(in_features=300,out_features=100)
        self.relu_ip2 = nn.ReLU(inplace=True)
        self.ip3 = t3.TTLinear(in_features=100,out_features=10)

    def forward(self,x):
        x = x.view(x.size(0), 28*28)
        x = self.ip1(x)
        x = self.relu_ip1(x)
        x = self.ip2(x)
        x = self.relu_ip2(x)
        x = self.ip3(x)
        return F.log_softmax(x,dim=1)

class Dense_Net(nn.Module):
    def __init__(self):
        super(Dense_Net, self).__init__()
        self.ip1 = nn.Linear(28*28, 300)
        self.relu_ip1 = nn.ReLU(inplace=True)
        self.ip2 = nn.Linear(300, 100)
        self.relu_ip2 = nn.ReLU(inplace=True)
        self.ip3 = nn.Linear(100, 10)
        return

    def forward(self, x):
        x = x.view(x.size(0), 28*28)
        x = self.ip1(x)
        x = self.relu_ip1(x)
        x = self.ip2(x)
        x = self.relu_ip2(x)
        x = self.ip3(x)

        return F.log_softmax(x,dim=1)


class PTT_OptNet(nn.Module):
    def __init__(self,init=None):
        super(PTT_OptNet,self).__init__()
        self.ip1 = t3.TTSolver(in_features=28*28,out_features=300)
        self.ip2 = t3.TTSolver(in_features=300,out_features=100)
        self.ip3 = t3.TTSolver(in_features=100,out_features=10)

    def forward(self,x):
        x = x.view(x.size(0), 28*28)
        x = self.ip1(x)
        x = self.ip2(x)
        x = self.ip3(x)
        return F.log_softmax(x,dim=1)


class FTT_OptNet(nn.Module):
    def __init__(self,init=None):
        super(FTT_OptNet,self).__init__()
        self.ip1 = t3.FTTSolver(in_features=28*28,out_features=300)
        self.ip2 = t3.FTTSolver(in_features=300,out_features=100)
        self.ip3 = t3.FTTSolver(in_features=100,out_features=10)

    def forward(self,x):
        x = x.view(x.size(0), 28*28)
        x = self.ip1(x)
        x = self.ip2(x)
        x = self.ip3(x)
        return F.log_softmax(x,dim=1)
