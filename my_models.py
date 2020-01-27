from __future__ import print_function
import torch
import torch.nn as nn
import os
import sys
import t3nsor as t3
import torch.nn.functional as F
import pdb
import torchvision.models as models
import model_wideresnet
from t3nsor import TensorTrainBatch

class normal_logistic(nn.Module):
    def __init__(self, settings):
        super(normal_logistic, self).__init__()
        self.ip = nn.Linear(32*32*3, 10)

    def forward(self, x):
        batch_size = x.shape[0]
        x = torch.reshape(x, [batch_size, 32*32*3])
        x = self.ip(x)
        out = F.softmax(x)
        return out


#last layer is an element wise linear layer instead of tt_to_dense procedure, intended to avoid gradient exploding
class IS_FTT_multi_layer_relu_l(nn.Module):
    def __init__(self, settings):
        super(IS_FTT_multi_layer_relu_l, self).__init__()
        self.ip1 = t3.FTT_Solver(in_features=177, out_features=2000,
                                     shape=[[3, 4, 8, 4, 8], [1, 2, 50, 20, 1]], tt_rank=3)
        self.ip2 = t3.FTT_Solver(in_features=2000, out_features=2000,
                                     shape=[[1, 2, 50, 20, 1], [1, 2, 50, 20, 1]], tt_rank=3)
        self.ip3 = t3.FTT_Solver(in_features=2000, out_features=2000,
                                     shape=[[1, 2, 50, 20, 1], [1, 2, 50, 20, 1]], tt_rank=3)
        self.ip4 = t3.FTT_Solver(in_features=2000, out_features=2000,
                                     shape=[[1, 2, 50, 20, 1], [1, 2, 50, 20, 1]], tt_rank=3)
        self.ip5 = t3.FTT_Solver(in_features=2000, out_features=2000,
                                     shape=[[1, 2, 50, 20, 1], [1, 2, 50, 20, 1]], tt_rank=3)
        self.ip6 = t3.FTT_Solver(in_features=2000, out_features=10, shape=[[1, 2, 50, 20, 1], [1, 2, 5, 1, 1]], tt_rank=3)
        #self.ip_ult_linear = nn.Linear(in_features=366, out_features=10)
        self.batch_size = settings.BATCH_SIZE
        self.tt_to_dense = t3.layers.tt_to_dense()

    def forward(self, x):
        #pdb.set_trace()
        #TODO: customize nn.DataParallel to simplify code, now we are doing repeat work of conversion
        #from dense to TensorTrainBatch

        x_1 = x[:,0:9].view(-1,3,3)
        x_2 = x[:,9:9+36].view(-1,3,4,3)
        x_3 = x[:,45:72+45].view(-1,3,8,3)
        x_4 = x[:,117:117+36].view(-1,3,4,3)
        x_5 = x[:,153:153+24].view(-1,3,8)

        x_1 = torch.unsqueeze(x_1, dim=1)
        x_1 = torch.unsqueeze(x_1, dim=1)
        x_2 = torch.unsqueeze(x_2, dim=2)
        x_3 = torch.unsqueeze(x_3, dim=2)
        x_4 = torch.unsqueeze(x_4, dim=2)
        x_5 = torch.unsqueeze(x_5, dim=-1)
        x_5 = torch.unsqueeze(x_5, dim=2)
        #pdb.set_trace()

        cov_list = []
        cov_list.append(x_1)
        cov_list.append(x_2)
        cov_list.append(x_3)
        cov_list.append(x_4)
        cov_list.append(x_5)

        x  = t3.TensorTrainBatch(cov_list)

        x = self.ip1(x)
        x = self.ip2(x)
        x = self.ip3(x)
        x = self.ip4(x)
        x = self.ip5(x)

        """last linear layer"""
        #re = [torch.reshape(tt_core, (int(self.batch_size/torch.cuda.device_count()), -1)) for tt_core in x.tt_cores]
        #x = torch.cat(re, dim=1)
        #x = self.ip_ult_linear(x)

        """last tt_to_dense layer"""
        x = self.ip6(x)
        x = self.tt_to_dense(x)
        x = torch.squeeze(x)
        return F.log_softmax(x, dim=1)


class IS_FTT_1_layer_relu(nn.Module):
    def __init__(self, settings):
        super(IS_FTT_1_layer_relu, self).__init__()
        self.ip1 = t3.FTT_Solver(in_features=177, out_features=2000000, shape=[[3, 4, 8, 4, 8], [10, 20, 50, 20, 10]], tt_rank=3)
        self.ip2 = t3.FTT_Solver(in_features=2000000, out_features=10, shape=[[10, 20, 50, 20, 10], [1, 2, 5, 1, 1]], tt_rank=3)
        self.to_dense = t3.layers.tt_to_dense()

    def forward(self, x):
        x = self.ip1(x)
        x = self.ip2(x)
        x = self.to_dense(x)
        x = torch.squeeze(x)
        #pdb.set_trace()
        return F.log_softmax(x, dim=1)


class Logistic(nn.Module):
    def __init__(self, settings):
        super(Logistic, self).__init__()
        self.linear = nn.Linear(177, 10)

    def forward(self, x):
        out = F.softmax(self.linear(x))
        return out


#add trainable parameter in SVD procedure, unstable and does not work
class learnable_tt_wide_resnet(nn.Module):
    def __init__(self, settings):
        super(learnable_tt_wide_resnet, self).__init__()
        self.batch_size = settings.BATCH_SIZE
        self.learnable_conversion = t3.layer_to_tt_tensor_learnable(shape=settings.TT_SHAPE, tt_rank=settings.TT_RANK, settings=settings)
        self.layer1 = t3.core_wise_linear(shape=settings.TT_SHAPE, tt_rank=settings.TT_RANK, in_channels=3, out_channels=32*32*3, settings=settings)
        #self.layer2 =  models.__dict__['wide_resnet50_2']()
        self.layer2 = model_wideresnet.WideResNet(depth=28, num_classes=10)
        #self.fc = nn.Linear(1000,10)
        self.shape = [self.batch_size, 3, 32, 32]

    def forward(self, x):
        x = self.learnable_conversion(x)
        x = self.layer1(x)
        x = torch.reshape(x, shape=self.shape)
        x = self.layer2(x)
        #x = self.fc(x)
        return x


class tt_input_wideresnet(nn.Module):
    def __init__(self, settings):
        super(tt_input_wideresnet, self).__init__()
        self.batch_size = settings.BATCH_SIZE
        self.layer1 = t3.core_wise_linear(shape=settings.TT_SHAPE, tt_rank=settings.TT_RANK, in_channels=3, out_channels=32*32*3, settings=settings)
        #self.layer2 =  models.__dict__['wide_resnet50_2']()
        self.layer2 = model_wideresnet.WideResNet(depth=28, num_classes=10)
        #self.fc = nn.Linear(1000,10)
        self.shape = [self.batch_size, 3, 32, 32]

    def forward(self, x):
        x = self.layer1(x)
        x = torch.reshape(x, shape=self.shape)
        x = self.layer2(x)
        #x = self.fc(x)
        return x


class important_sketching_input_wideresnet(nn.Module):
    def __init__(self, settings):
        super(important_sketching_input_wideresnet, self).__init__()
        self.batch_size = settings.BATCH_SIZE

        self.layer1 = nn.Linear(177, 32*32*3)
        #self.layer2 =  models.__dict__['wide_resnet50_2']()
        self.layer2 = model_wideresnet.WideResNet(settings, depth=28, num_classes=10)
        #self.fc = nn.Linear(1000,10)
        self.shape = [self.batch_size, 3, 32, 32]

    def forward(self, x):

        x = self.layer1(x)
        x = torch.reshape(x, shape=self.shape)
        x = self.layer2(x)
        #x = self.fc(x)
        return x


#single-layer-perceptron
class slp(nn.Module):
    def __init__(self, settings):
        super(slp, self).__init__()
        self.batch_size = settings.BATCH_SIZE
        self.fc1 = nn.Linear(3*32*32, 10)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        #pdb.set_trace()
        x = x.view(self.batch_size,-1)
        x = self.fc1(x)
        #x = self.fc2(x)
        #x = self.fc3(x)
        return x


#core_wise_linear
class cw_fc_net(nn.Module):
    def __init__(self,settings):
        super(cw_fc_net,self).__init__()
        #self.to_tt = t3.layers_to_tt_tensor(settings)
        self.fc = t3.core_wise_linear(shape=settings.TT_SHAPE, tt_rank=settings.TT_RANK, in_channels=3, out_channels=10, settings=settings)
        self.fc2 = nn.Linear(100, 10)

    def forward(self,x):
        #x = self.to_tt(x)
        x = self.fc(x)
        #x = self.fc2(x)
        return x


# class pre_mani_net(nn.Module):
#     def __init__(self,settings):
#         super(pre_mani_net,self).__init__()
#         self.to_tt = t3.layer_to_tt_tensor(settings)
#         self.conv1 = t3.TTConv(settings.TT_SHAPE,in_channels=1,out_channels=16)
#         # self.conv1 = nn.Conv2d(3,6,5)
#         # self.pool = nn.MaxPool2d(2,2)
#         # self.conv2 = nn.Conv2d(6,16,5)
#         # self.conv3 = nn.Conv2d(16,3,5)
#         self.fc1 = t3.TTFC(settings.TT_SHAPE,tt_rank=settings.TT_RANK,in_channels=16)
#         #self.solver1 = t3.TTSolver(in_features=16,out_features=120)
#         #self.solver2= t3.TTSolver(in_features=120,out_features=84)
#         self.fc2 = t3.TTLinear(in_features=16,out_features=16)
#         self.relu_ip1 = nn.ReLU(inplace=True)
#         self.fc3 = t3.TTLinear(in_features=16,out_features=84)
#         self.relu_ip2 = nn.ReLU(inplace=True)
#         self.fc4 = nn.Linear(84,10)
#
#     def forward(self, x):
#         x = self.to_tt(x)
#         x = self.conv1(x)
#         # x = self.conv2(x)
#         # x = self.conv3(x)
#         #x = self.conv4(x)
#         #x = self.conv5(x)
#         x = self.fc1(x)
#         #x = self.solver1(x)
#         #x = self.solver2(x)
#         x = self.fc2(x)
#         x = self.relu_ip1(x)
#         x = self.fc3(x)
#         x = self.relu_ip2(x)
#         x = self.fc4(x)
#         #return F.log_softmax(x, dim=1)
#         return x


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
        #return F.log_softmax(x, dim=1)
        return x


class pre_FTT_Net(nn.Module):
    def __init__(self, settings):
        super(pre_FTT_Net, self).__init__()
        self.to_tt_matrix = t3.layer_to_tt_matrix(settings)
        self.ip1 = t3.FTTLinear(in_features=512, out_features=10)
        self.to_dense = t3.layers.tt_to_dense()

    def forward(self, x):
        x = self.to_tt_matrix(x)
        x = self.ip1(x)
        x = self.to_dense(x)
        return F.log_softmax(x, dim=1)


class IS_FTT_Logistic(nn.Module):
    def __init__(self, settings):
        super(IS_FTT_Logistic, self).__init__()
        self.ip1 = t3.FTTLinear(in_features=177, out_features=10)
        self.ip2 = nn.Linear(in_features=666, out_features=10)
        self.to_dense = t3.layers.tt_to_dense()

    def forward(self, x):
        x = self.ip1(x)
        #pdb.set_trace()
        re = [torch.reshape(tt_core, (64,-1)) for tt_core in x.tt_cores]
        x = torch.cat(re,dim=1)
        x = self.ip2(x)
       # pdb.set_trace()
       # x = self.to_dense(x)

        #x = torch.squeeze(x, dim=1)

        return x



#tensorizing neural network, Alexander et al 2015
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

        return F.log_softmax(x, dim=1)


class PTT_OptNet(nn.Module):
    def __init__(self, init=None):
        super(PTT_OptNet, self).__init__()
        self.ip1 = t3.TTSolver(in_features=28*28, out_features=300)
        self.ip2 = t3.TTSolver(in_features=300, out_features=100)
        self.ip3 = t3.TTSolver(in_features=100, out_features=10)

    def forward(self, x):
        x = x.view(x.size(0), 28*28)
        x = self.ip1(x)
        x = self.ip2(x)
        x = self.ip3(x)
        return F.log_softmax(x, dim=1)


class FTT_OptNet(nn.Module):
    def __init__(self, init=None):
        super(FTT_OptNet, self).__init__()
        self.ip1 = t3.FTT_Solver(in_features=28*28, out_features=300)
        self.ip2 = t3.FTT_Solver(in_features=300, out_features=100)
        self.ip3 = t3.FTT_Solver(in_features=100, out_features=10)

    def forward(self, x):
        x = x.view(x.size(0), 28*28)
        x = self.ip1(x)
        x = self.ip2(x)
        x = self.ip3(x)
        return F.log_softmax(x, dim=1)
