from __future__ import print_function
import torch
import torch.nn as nn
import os
import sys
import t3nsor as t3
import torch.nn.functional as F
import pdb
import torchvision.models as models
from model_wideresnet import *
from t3nsor import TensorTrainBatch
from t3nsor.utils import *

class linear_exp(nn.Module):
    def __init__(self, settings, num_classes=10):
        super(linear_exp, self).__init__()
        self.fc1 = nn.Linear(4096,10)
        self.exp = t3.exp_machine(settings=settings, weight_channel=num_classes, shape=10*[2])
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.exp(x)
        x = self.fc2(x)
        #pdb.set_trace()
        return x


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*2*2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        #self.exp_layer = t3.exp_machine(settings=settings, weight_channel=num_classes, shape=10*[2])
        #self.final_fc = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256*2*2)
        x = self.classifier(x)
        return x


class AlexNet_exp(nn.Module):
    def __init__(self, settings, num_classes=10):
        super(AlexNet_exp, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        #self.avgpool=nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*2*2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.exp_layer = t3.exp_machine(settings=settings, weight_channel=num_classes, shape=10*[2])
        self.final_fc = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256*2*2)
        x = self.classifier(x)
        x = self.exp_layer(x)
        x = self.final_fc(x)
        return x

class all_interaction_linear(nn.Module):
    def __init__(self, settings, weight_channel=1):
        super(all_interaction_linear, self).__init__()
        self.exp = t3.exp_machine(settings=settings, weight_channel=weight_channel)
    def forward(self, x):
        out = self.exp(x)


class all_interaction(nn.Module):
    def __init__(self, settings, weight_channel=10):
        super(all_interaction, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.exp_interact = t3.exp_machine(settings=settings, weight_channel=weight_channel, shape=10*[2])
        self.fc1 = nn.Linear(weight_channel, 120)
        self.fc11 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc11(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = self.exp_interact(out)
        return out


class end_to_end(nn.Module):
    def __init__(self, settings, important_directions):
        super(end_to_end, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 15, 5)
        self.layer = important_sketching_input_wideresnet(settings)
        self.important_directions = important_directions
        self.settings = settings

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)

        #output shape: [batch_size, 15, 5, 5]
        #compute cov matrix
        beta = 0.3
        batch_size = out.shape[0]
        mean = torch.unsqueeze(torch.sum(out.view(batch_size, 15, 25), 2), dim=2)
        assert(mean.shape[0] == batch_size and mean.shape[1] == 15)

        batch_cov_matrix = 1/25 * torch.matmul(out.view(batch_size, 15, 25) - torch.cat(25*[mean], dim=2), (out.view(batch_size, 15, 25)-torch.cat(25*[mean], dim=2)).view(batch_size, 25, 15))
        batch_cov_matrix += beta**2*mean.matmul(mean.transpose(1, 2))
        upper_part = torch.cat([batch_cov_matrix, beta*mean], dim=2)
        lower_part = torch.cat([beta*mean.transpose(1, 2), torch.ones(batch_size, 1, 1).to(mean.device)], dim=2)
        final = torch.cat([upper_part, lower_part], dim=1)
        #print('final shape:', final.shape)
        #pdb.set_trace()
        projected = important_sketching(final, self.important_directions, self.settings)
        out = self.layer(projected)
        return out


class LeNet_partial(nn.Module):
    def __init__(self):
        super(LeNet_partial, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 15, 5)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)

        #output shape: [batch_size, 15, 5, 5]
        #compute cov matrix
        beta = 0.3
        batch_size = out.shape[0]
        mean = torch.unsqueeze(torch.sum(out.view(batch_size, 15, 25), 2), dim=2)
        assert(mean.shape[0] == batch_size and mean.shape[1] == 15)

        batch_cov_matrix = 1/25 * torch.matmul(out.view(batch_size, 15, 25) - torch.cat(25*[mean], dim=2), (out.view(batch_size, 15, 25)-torch.cat(25*[mean], dim=2)).view(batch_size, 25, 15))
        batch_cov_matrix += beta**2*mean.matmul(mean.transpose(1, 2))
        upper_part = torch.cat([batch_cov_matrix, beta*mean], dim=2)
        lower_part = torch.cat([beta*mean.transpose(1, 2), torch.ones(batch_size, 1, 1).to(mean.device)], dim=2)
        final = torch.cat([upper_part, lower_part], dim=1)
        #print('final shape:', final.shape)
        return final


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 15, 5)
        self.fc1 = nn.Linear(15*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


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
class IS_FTT_multi_layer_relu(nn.Module):
    def __init__(self, settings):
        super(IS_FTT_multi_layer_relu, self).__init__()
        self.ip1 = t3.FTT_Solver(shape=[[3, 4, 8, 4, 8], [4, 5, 10, 5, 4]], tt_rank=3)
        self.ip2 = t3.FTT_Solver(shape=[[4, 5, 10, 5, 4], [4, 5, 10, 5, 4]], tt_rank=3)
        self.ip3 = t3.FTT_Solver(shape=[[4, 5, 10, 5, 4], [4, 5, 10, 5, 4]], tt_rank=3)
        self.ip4 = t3.FTT_Solver(shape=[[4, 5, 10, 5, 4], [4, 5, 10, 5, 4]], tt_rank=3)
        self.ip5 = t3.FTT_Solver(shape=[[4, 5, 10, 5, 4], [4, 5, 10, 5, 4]], tt_rank=3)
        self.ip6 = t3.FTT_Solver(shape=[[4, 5, 10, 5, 4], [1, 2, 5, 1, 1]], tt_rank=3)
        #self.ip_ult_linear = nn.Linear(in_features=366, out_features=10)
        self.batch_size = settings.BATCH_SIZE
        self.tt_to_dense = t3.layers.tt_to_dense()
        self.settings = settings

    def forward(self, x):
        #pdb.set_trace()
        #TODO: customize nn.DataParallel to simplify code, now we are doing repeat work of conversion
        #from dense to TensorTrainBatch

        x_1 = x[:, 0:9].view(-1, 3, 3)
        x_2 = x[:, 9:9+36].view(-1, 3, 4, 3)
        x_3 = x[:, 45:72+45].view(-1, 3, 8, 3)
        x_4 = x[:, 117:117+36].view(-1, 3, 4, 3)
        x_5 = x[:, 153:153+24].view(-1, 3, 8)

        x_1 = torch.unsqueeze(x_1, dim=1)
        x_1 = torch.unsqueeze(x_1, dim=1)
        x_2 = torch.unsqueeze(x_2, dim=2)
        x_3 = torch.unsqueeze(x_3, dim=2)
        x_4 = torch.unsqueeze(x_4, dim=2)
        x_5 = torch.unsqueeze(x_5, dim=-1)
        x_5 = torch.unsqueeze(x_5, dim=2)

        cov_list = []
        cov_list.append(x_1)
        cov_list.append(x_2)
        cov_list.append(x_3)
        cov_list.append(x_4)
        cov_list.append(x_5)

        x = t3.TensorTrainBatch(cov_list)
        #pdb.set_trace()

        svd_net_loss = 0
        svd_matrix_list = []
        x = self.ip1(x)
        #x, m1, m2 = self.ip1(x)
        #svd_matrix_list.append(m1)
        #svd_net_loss += m2
        #svd_matrix_list.append(m2)
        x = self.ip2(x)
        #x, m1, m2 = self.ip2(x)
        #svd_matrix_list.append(m1)
        #svd_net_loss += m2
        #svd_matrix_list.append(m2)
        x = self.ip3(x)
        #x, m1, m2 = self.ip3(x)
        #svd_matrix_list.append(m1)
        #svd_net_loss += m2
        #svd_matrix_list.append(m2)
        x = self.ip4(x)
        #x, m1, m2 = self.ip4(x)
        #svd_matrix_list.append(m1)
        #svd_net_loss += m2
        #svd_matrix_list.append(m2)
        x = self.ip5(x)
        # x, m1, m2 = self.ip5(x)
        # svd_matrix_list.append(m1)
        # svd_net_loss += m2
        #svd_matrix_list.append(m2)
        """last linear layer"""
        #re = [torch.reshape(tt_core, (int(self.batch_size/torch.cuda.device_count()), -1)) for tt_core in x.tt_cores]
        #x = torch.cat(re, dim=1)
        #x = self.ip_ult_linear(x)

        """last tt_to_dense layer"""
        x = self.ip6(x)
        # x, m1, m2 = self.ip6(x)
        # svd_matrix_list.append(m1)
        # svd_net_loss += m2
        #svd_matrix_list.append(m2)
        x = self.tt_to_dense(x)
        x = torch.squeeze(x)
        #pdb.set_trace()

        #print("\tIn Model: input size", x.size())

        """DataParallel cannot take care of model which output nested list(they can handle list?, tuple?, tensor, dict)
         , here we have 4*12 primary element to output, which is used to
         construct loss function. So we wrap the loss function within forward, and output a 2 element tuple, instead of 48+1"""
        # alpha = self.settings.ALPHA
        # beta = self.settings.BETA
        # loss_orth = 0
        # count = 0
        # for svd_matrix in svd_matrix_list:
        #     for curr_core in svd_matrix:
        #         # pdb.set_trace()
        #         assert (len(curr_core.shape) == 2)
        #         #for row_idx in range(min(curr_core.shape[0], curr_core.shape[1])):
        #         #    count += 1
        #         #    if alpha * torch.sum(torch.abs(curr_core[row_idx, :] / curr_core[row_idx, row_idx])) > 10:
        #         #        print("adding a huge loss!")
        #         #    loss = loss + alpha * torch.sum(torch.abs(curr_core[row_idx, :] / curr_core[row_idx, row_idx])) ** 2
        #         #diff = torch.norm(torch.mm(curr_core.permute(1, 0), curr_core) - torch.ones(curr_core.shape[1], curr_core.shape[1]).to('cuda'))
        #         #print("distance with Identity is :", diff)
        #         loss_orth += alpha * torch.norm(torch.mm(curr_core.permute(1, 0), curr_core) - torch.ones(curr_core.shape[1], curr_core.shape[1]).to('cuda'))
        #         count += 1

        return F.log_softmax(x, dim=1)#, torch.div(loss_orth, count), beta*svd_net_loss


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

        self.layer1 = nn.Linear(160, 32*32*3)
        #self.layer2 =  models.__dict__['wide_resnet50_2']()
        self.layer2 = WideResNet(depth=28, num_classes=10)
        #self.fc = nn.Linear(1000,10)

    def forward(self, x):

        x = self.layer1(x)
        x = torch.reshape(x, [-1, 3, 32, 32])
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
