import torch
import numpy as np
import torch.nn as nn
import t3nsor as t3
import math
from t3nsor.utils import cayley
from t3nsor import TensorTrainBatch
from t3nsor import TensorTrain

import pdb

#only work for input channel 3, TODO: other channel
class to_tt(nn.Module):
    def __init__(self,settings):
        super(to_tt,self).__init__()
        self.batch_size = settings.BATCH_SIZE
        self.tt_shape = settings.TT_SHAPE
        self.tt_rank = settings.TT_RANK
        self.settings = settings
    def forward(self,input):
        input = input.view(self.batch_size,1,8,64)
        #pdb.set_trace()
        #convert input from dense format to TT format, specificaly, TensorTrainBatch
        return t3.input_to_tt(input,self.settings)

#input: TensorTrainBatch1
class TTFC(nn.Module):
    def __init__(self,shape,tt_rank=4,in_channels=120,init=None):
        super(TTFC,self).__init__()
        #example: shape = [7,8,9] d=3 n1=7 n2=8 n3=9
        self.ndims = len(shape)
        self.jj = int(np.sum(shape))
        self.shape = shape
        self.in_channels = in_channels
        self.init_shape = [8,8,8,8]
        #self.init_shape = [4,5,4,5]
        if init is None:
            #init = t3.glorot_initializer(self.init_shape, tt_rank=tt_rank)
            init = t3.tensor_with_random_cores(self.init_shape,tt_rank = tt_rank)
        self.weight = init.to_parameter()
        self.parameters = self.weight.parameter

    def forward(self,input):
        #print('weight shape:',self.weight)
        #list length: in_channels, each element in list will contribute to one column of the output [batch_size, in_channels]
        list_in_list = [i.tt_cores for i in input]

        # stacked_core_list = []
        # for core_iter in range(0,self.ndims):
        #     temp_list = []
        #     for i in range(0,in_channels):
        #         temp_list += re_organized[i + core_iter*in_channels]
        #     stacked_core_list += torch.stack(*temp_list,dim=0)
        norm_matrix = 0
        for tt_batch_iter in range(0,self.in_channels):
            curr_batch_cores = list_in_list[tt_batch_iter]
            #curr_batch_cores is a list of tt_cores, each tt_core is 4 dimension: batch_size, r_1, n_1, r_2
            count_2_norm = 0
            for core_iter in range(0,len(curr_batch_cores)-1):

                #weight in exactly same TT format as input
                input_curr_core = curr_batch_cores[core_iter]
                weight_curr_core = self.weight.tt_cores[core_iter]
                n = input_curr_core.shape[2]
                for slice_iter in range(0,n):
                    #out.shape: batch_size, r, r
                    out = torch.einsum('jk,bji->bki', weight_curr_core[:,slice_iter,:], input_curr_core[:,:,slice_iter,:])
                    out_t = 1/2*(out.transpose(1,2)-out)

                    # print(out_t.shape)
                    # pdb.set_trace()

                    # print('out shape:',out.shape)
                    # if max(torch.norm(out,dim=(1,2)))<1e-8:
                    #     print('zero value in norm!!!!!!')
                    #gradient of zero norm is nan, need to add an eps
                    eps = 1e-5
                    count_2_norm = count_2_norm + torch.norm(out_t+eps*torch.ones(out.shape).to(out.device),dim=(1,2))**2
            #norm_list will be a list of torch tensors
            #print('norm_matrix shape222:',norm_matrix)
            count_2_norm_sqrt = torch.sqrt(count_2_norm + eps)
            if tt_batch_iter == 0:
                #print('count_2_norm shape:',count_2_norm.shape)
                norm_matrix = torch.unsqueeze(count_2_norm_sqrt,dim=1)
            else:
                #pdb.set_trace()
                norm_matrix = torch.cat((norm_matrix,torch.unsqueeze(count_2_norm_sqrt,dim=1)),dim=1)

                #TODO: cat

        #print(norm_matrix.view(out.shape[0],-1))
        #pdb.set_trace()
        return norm_matrix.view(out.shape[0],-1)
        #shape of output: [batch_size,in_channels]


class TTConv(nn.Module):
    #N is in_channels
    #in_channels=3 means input will be 3 TensorTrainBatch, out_channels=84 means output will be 84 TensorTrainBatch
    def __init__(self,shape,in_channels=3,out_channels=120):
        super(TTConv,self).__init__()
        #example: shape = [4,8,4,8] d=4 n1=4 n2=8 n3=4, n4=8
        self.ndims = len(shape)
        self.jj = int(np.sum(shape))
        #initilize the same as nn.Linear
        self.weight = nn.Parameter(torch.Tensor(in_channels, self.jj, out_channels))
        self.shape = shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        #self.weight.data.constant_(1/self.N)
        #nn.init.constant_(self.weight, 1/self.in_channels)

    #input:list(TensorTrainBatch1,TensorTrainBatch2,TensorTrainBatch3)
    def forward(self,input):
        list_in_list = [i.tt_cores for i in input]
        re_organized = []
        for core_iter in range(0,self.ndims):
            temp = [i[core_iter] for i in list_in_list]
            re_organized += temp
        #print('re_organized length:',len(re_organized))
        #re_organized will be a list: [TenosorTrainBatch_1.cores[0],TensorTrainBatch_2.cores[0],...,TensorTrainBatch_N.cores[0],
        #                              TensorTrainBatch_1.cores[1], TensorTrainBatch_2.cores[1],...,TensorTrainBatch_N.cores[1],
        #                               ......
        #
        #cores: (batch_size, r_1, slice_iter, r_2)
        #Use TensorTrainBatch_1.cores[0], TensorTrainBatch_2.cores[0],... to construct output_TensorTrainBatch.cores[0]
        #stack these cores to becomes a big tensor: [in_channels, batch_size, r_1, slice_iter, r_2]
        stacked_core_list = []
        for core_iter in range(0,self.ndims):
            temp_list = []
            for i in range(0,self.in_channels):
                temp_list.append(re_organized[i + core_iter*self.in_channels])
            stacked_core_list.append(torch.stack(temp_list,dim=0))
        #stacked_core_list will be [stacked_1,stacked_2,stracked_3,stracked_4]

        #check duplication: not duplicate until here

        #apply logm
        for i in range(1,self.ndims-1):
            stacked_core_list[i] = 1/2*(stacked_core_list[i].transpose(2,4) - stacked_core_list[i])
            #pdb.set_trace()

        out_slice_list = []
        cat_slice = []
        slice_cumulative_count = 0
        for core_iter in range(0,self.ndims):
            if core_iter == 0:
                for slice_iter in range(0,self.shape[core_iter]):
                    out_slice_list.append(torch.unsqueeze(torch.einsum('io,ibkj->obkj', self.weight[:,slice_cumulative_count,:], stacked_core_list[core_iter][:,:,:,slice_iter,:]),dim=3))
                    slice_cumulative_count += 1
                cat_slice.append(torch.cat(out_slice_list,dim=3))
                out_slice_list = []
            elif core_iter == self.ndims-1:
                for slice_iter in range(0,self.shape[core_iter]):
                    out_slice_list.append(torch.unsqueeze(torch.einsum('io,ibkj->obkj', self.weight[:,slice_cumulative_count,:], stacked_core_list[core_iter][:,:,:,slice_iter,:]),dim=3))
                    slice_cumulative_count += 1
                cat_slice.append(torch.cat(out_slice_list,dim=3))
                out_slice_list = []
            else:
                #pdb.set_trace()
                for slice_iter in range(0,self.shape[core_iter]):
                    out_slice_list.append(torch.unsqueeze(cayley(torch.einsum('io,ibkj->obkj', self.weight[:,slice_cumulative_count,:], stacked_core_list[core_iter][:,:,:,slice_iter,:])),dim=3))
                    slice_cumulative_count += 1
                #cat_slice already duplicated!!!
                cat_slice.append(torch.cat(out_slice_list,dim=3))
                out_slice_list = []

        #shape of element in cat_slice: [out_channels,batch_size,r_1,slice_iter,r_2]
        #split along the out_channels dimension
        #splited_list is a list of tuple
        splited_list = [torch.split(curr_core,split_size_or_sections=1,dim=0) for curr_core in cat_slice]
        re_organized_splited_list = []
        for iter in range(0,self.out_channels):
            temp = [torch.squeeze(i[iter],0) for i in splited_list]
            re_organized_splited_list += temp
        ttbatch_list = []
        for i in range(0,self.out_channels):
            ttbatch_list.append(TensorTrainBatch(re_organized_splited_list[i*self.ndims:i*self.ndims+self.ndims:1],convert_to_tensors=False))
        #pdb.set_trace()
        return ttbatch_list


        # i = -1
        # j = 0
        #
        # #M is the list of batched slices
        # #construct M
        # M = [] #len:self.jj*N
        # for tt_batch in input
        #     i = i + 1
        #     cores = tt_batch.tt_cores
        #     #cores is a list of tt_cores, each tt_core is 4 dimension: batch_size, r_1, n_1, r_2
        #
        #     #first core: keep same
        #     for i in range(0,cores[0].shape[2]):
        #         M.append(cores[0][:,:,i,:])
        #
        #     #middle cores
        #     for tt_core in cores:
        #         if tt_core.shape[1] == 1 or tt_core.shape[3] == 1:
        #             continue
        #         n = tt_core.shape[2]
        #         for temp in range(0,n):
        #             skew_sym_part = 1/2*(tt_core[:,:,temp,:].transpose(1,3)-tt_core[:,:,temp,:])
        #             M.append(skew_sym_part * self.weight[i][j+temp])
        #         j = j + n
        #
        #      #last core: keep same
        #     for i in range(0,cores[-1].shape[2]):
        #          M.append(cores[-1][:,:,i,:])
        #
        # #re-organize M to get output
        # #M = [TensorTrainBatch1.tt_core1.slice1, ...]
        # #consturct ouput TensorTrainBatch TT cores
        #
        # M_new = [] #len:self.jj
        # #first core, keep same
        # for i in range(0,self.shape[0]):
        #     M_new.append(M[i])
        #
        # #middle cores
        # for i in range(self.shape[0],self.jj-self.shape[-1]):
        #     M_new.append(cayley(M[i] + M[i+self.jj] + M[i + 2*self.jj]))
        #
        # #last core: keep same
        # for i in range(self.jj- self.shape[-1], self.jj):
        #     M_new.append(M[i])
        #
        # cumulate_index = 0
        # new_tt_cores = []
        # for core_index in range(0,self.dims):
        #     curr_core = []
        #     for i in range(self.shape[core_index]):
        #         if i == 0:
        #             curr_core = torch.unsqueeze(M_new[cumulate_index],2)
        #         else:
        #             curr_core = torch.cat((curr_core, M_new[cumulate_index + i]),dim=2)
        #     cumulate_index = cumulate_index + self.shape[core_index]
        #     new_tt_cores = new_tt_cores.append(curr_core)
        # return TensorTrainBatch(new_tt_cores, convert_to_tensors=False)


class TTLinear(nn.Module):
    def __init__(self, in_features=None, out_features=None, bias=True, init=None, shape=None,
                 auto_shapes=True, d=3, tt_rank=8, stddev=None, auto_shape_mode='ascending',
                 auto_shape_criterion='entropy',
                 ):

        super(TTLinear, self).__init__()

        if auto_shapes:
            if in_features is None or out_features is None:
                raise ValueError("Shape is not specified")

            in_quantization = t3.utils.auto_shape(
                in_features, d=d, criterion=auto_shape_criterion, mode=auto_shape_mode)
            out_quantization = t3.utils.auto_shape(
                out_features, d=d, criterion=auto_shape_criterion, mode=auto_shape_mode)

            shape = [in_quantization, out_quantization]

        if init is None:
            if shape is None:
                raise ValueError(
                    "if init is not provided, please specify shape, or set auto_shapes=True")
        else:
            shape = init.raw_shape

        if init is None:
            init = t3.glorot_initializer(shape, tt_rank=tt_rank)

        self.shape = shape
        self.weight = init.to_parameter()
        self.parameters = self.weight.parameter
        #self.weight_t = t3.transpose(self.weight,convert_to_tensors=False)

        if bias:
            self.bias = torch.nn.Parameter(1e-2 * torch.ones(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        #Hint: put all forward pass computation in forward.
        #Wrong thing: if we put t3.transpose computation in init, model.cuda() cannot
        #transform output of transpose to cuda because it is not claimed to be nn.Parameter
        #weight_t = self.weight_t
        weight_t = t3.transpose(self.weight,convert_to_tensors=False)
        #print('weight_t device',weight_t.tt_cores[0].device)
        #print('weight device',self.weight.tt_cores[0].device)
        x_t = x.transpose(0, 1)
        if self.bias is None:
            temp=t3.tt_dense_matmul(weight_t, x_t).transpose(0, 1)
            return temp.to(x.device)
        else:
            #error occur!!!
            temp = t3.tt_dense_matmul(weight_t, x_t).transpose(0, 1) + self.bias
            #print('x device:',x.device)
            return temp.to(x.device)


class TTSolver(nn.Module):
    def __init__(self,
                in_features=None,
                out_features=None,
                bias=True,
                init=None,
                shape=None,
                auto_shapes=True,
                d=3,
                tt_rank=8,
                iter_num=2,
                l=1,
                s=-1,
                epsilon=1,
                auto_shape_mode='ascending',
                auto_shape_criterion='entropy'):

        super(TTSolver, self).__init__()

        if auto_shapes:
            print('auto_shape working...')
            if in_features is None or out_features is None:
                raise ValueError("Shape is not specified")

            in_quantization = t3.utils.auto_shape(
            in_features, d=d, criterion=auto_shape_criterion, mode=auto_shape_mode)
            out_quantization = t3.utils.auto_shape(
            out_features, d=d, criterion=auto_shape_criterion, mode=auto_shape_mode)

            shape = [in_quantization, out_quantization]

        if init is None:
            if shape is None:
                raise ValueError(
                        "if init is not provided, please specify shape, or set auto_shapes=True")
        else:
            shape = init.raw_shape

        if init is None:
            init = t3.glorot_initializer(shape, tt_rank=tt_rank)

        self.shape = shape
        self.weight = init.to_parameter()
        self.parameters = self.weight.parameter
        self.weight_t = t3.transpose(self.weight)
        self.iter_num = iter_num
        self.L = l
        self.S = s
        self.epsilon = epsilon
        self.out_features = out_features

        if bias:
            self.bias = torch.nn.Parameter(1e-2 * torch.ones(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        weight_t = t3.transpose(self.weight,False)
        #print('wight_t shape:', weight_t.shape)
        x_t = x.transpose(0, 1)
        #print('x_t shape:',x_t.shape)

        if self.bias is None:
            return t3.tt_dense_matmul(weight_t, x_t).transpose(0, 1)
        else:
            L = self.L
            S = self.S
            if self.iter_num == 1:
                if torch.norm(x) > self.epsilon:
                    return 1/L*(t3.tt_dense_matmul(weight_t,x_t).transpose(0,1)+self.bias)
                else:
                    return 1/L*(t3.tt_dense_matmul(weight_t,x_t).transpose(0,1)+self.bias) + S*torch.ones(self.out_features)
            else:
            # iter_num > 1
                if torch.norm(x) > self.epsilon:
                    x_l = 1/L*(t3.tt_dense_matmul(weight_t,x_t).transpose(0,1)+self.bias)
                else:
                    x_l = 1/L*(t3.tt_dense_matmul(weight_t,x_t).transpose(0,1)+self.bias) + S*torch.ones(self.out_features).to(x_t.device)
                #print('x_l shape:', x_l.shape)
            for i in range(1,self.iter_num):
                if torch.norm(x_l) > self.epsilon:
                    x_l = x_l - 1/L*t3.tt_dense_matmul(t3.tt_tt_mul(weight_t,self.weight),x_l.t()).transpose(0,1) + 1/L*t3.tt_dense_matmul(weight_t, x_t).transpose(0, 1) + self.bias
                else:
                    print('activated!!!')
                    x_l = x_l - 1/L*t3.tt_dense_matmul(t3.tt_tt_mul(weight_t,self.weight),x_l.t()).transpose(0,1) + 1/L*t3.tt_dense_matmul(weight_t, x_t).transpose(0, 1) + self.bias + S*torch.ones(self.out_features).to(x_t.device)
            return x_l



# class FTTLinear(nn.Module):
#     def __init__(self, in_features=None, out_features=None, bias=True, init=None, shape=None,
#                  auto_shapes=True, d=3, tt_rank=8, stddev=None, auto_shape_mode='ascending',
#                  auto_shape_criterion='entropy',
#                  ):
#
#         super(FTTLinear, self).__init__()
#
#         if auto_shapes:
#             if in_features is None or out_features is None:
#                 raise ValueError("Shape is not specified")
#
#             in_quantization = t3.utils.auto_shape(
#                 in_features, d=d, criterion=auto_shape_criterion, mode=auto_shape_mode)
#             out_quantization = t3.utils.auto_shape(
#                 out_features, d=d, criterion=auto_shape_criterion, modse=auto_shape_mode)
#
#             shape = [in_quantization, out_quantization]
#
#         if init is None:
#             if shape is None:
#                 raise ValueError(
#                     "if init is not provided, please specify shape, or set auto_shapes=True")
#         else:
#             shape = init.raw_shape
#
#         if init is None:
#             init = t3.glorot_initializer(shape, tt_rank=tt_rank)
#
#         self.shape = shape
#         self.weight = init.to_parameter()
#         self.parameters = self.weight.parameter
#         #put all forward computation in forward method, for .cuda()
#         #self.weight_t = t3.transpose(self.weight,convert_to_tensors=False)
#
#         bias_shape = init.raw_shape
#         if bias:
#             self.bias = torch.nn.Parameter(1e-2 * torch.ones(out_features))
#         else:
#             self.register_parameter('bias', None)
#
#     def forward(self, x):
#         #x will be a TensorTrainBatch
#
#
#         #Hint: put all forward pass computation in forward.
#         #Wrong thing: if we put t3.transpose computation in init, model.cuda() cannot
#         #transform output of transpose to cuda because it is not claimed to be nn.Parameter
#         #weight_t = self.weight_t
#         weight_t = t3.transpose(self.weight,convert_to_tensors=False)
#         #print('weight_t device',weight_t.tt_cores[0].device)
#         #print('weight device',self.weight.tt_cores[0].device)
#         x_t = x.transpose(0, 1)
#         if self.bias is None:
#             temp = t3.transpose(t3.tt_tt_matmul(weight_t, x_t))
#             return temp.to(x.device)
#         else:
#             temp = t3.add(t3.transpose(t3.tt_tt_matmul(weight_t, x_t)), self.bias)
#             #print('x device:',x.device)
#             return temp.to(x.device)
#
#
#
# class FTT_Solver(nn.Module):
#     def __init__(self,
#                 in_features=None,
#                 out_features=None,
#                 bias=True,
#                 init=None,
#                 shape=None,
#                 auto_shapes=True,
#                 d=3,
#                 tt_rank=8,
#                 iter_num=4,
#                 l=1,
#                 s=-1,
#                 epsilon=1,
#                 auto_shape_mode='ascending',
#                 auto_shape_criterion='entropy'):
#
#         super(FTTSolver, self).__init__()
#
#         if auto_shapes:
#             print('auto_shape working...')
#             if in_features is None or out_features is None:
#                 raise ValueError("Shape is not specified")
#
#             in_quantization = t3.utils.auto_shape(
#             in_features, d=d, criterion=auto_shape_criterion, mode=auto_shape_mode)
#             out_quantization = t3.utils.auto_shape(
#             out_features, d=d, criterion=auto_shape_criterion, mode=auto_shape_mode)
#
#             shape = [in_quantization, out_quantization]
#
#         if init is None:
#             if shape is None:
#                 raise ValueError(
#                         "if init is not provided, please specify shape, or set auto_shapes=True")
#         else:
#             shape = init.raw_shape
#
#         if init is None:
#             init = t3.glorot_initializer(shape, tt_rank=tt_rank)
#
#         self.shape = shape
#         self.weight = init.to_parameter()
#         self.parameters = self.weight.parameter
#         self.iter_num = iter_num
#         self.L = l
#         self.S = s
#         self.epsilon = epsilon
#         self.out_features = out_features
#
#         if bias:
#             self.bias = torch.nn.Parameter(1e-2 * torch.ones(out_features))
#         else:
#             self.register_parameter('bias', None)
#
#     def forward(self, x):
#         #x is in TT matrix format
#         weight_t = t3.transpose(self.weight,False)
#         #print('wight_t shape:', weight_t.shape)
#         x_t = t3.transpose(x,convert_to_tensors=False)
#         #print('x_t shape:',x_t.shape)
#
#         if self.bias is None:
#             print('not implemented error')
#         else:
#             L = self.L
#             S = self.S
#             if self.iter_num == 1:
#                 if torch.norm(x) > self.epsilon:
#                     return 1/L*(t3.tt_dense_matmul(weight_t,x_t).transpose(0,1)+self.bias)
#                 else:
#                     return 1/L*(t3.tt_dense_matmul(weight_t,x_t).transpose(0,1)+self.bias) + S*torch.ones(self.out_features)
#             else:
#             # iter_num > 1
#                 if torch.norm(x) > self.epsilon:
#                     x_l = 1/L*(t3.tt_dense_matmul(weight_t,x_t).transpose(0,1)+self.bias)
#                 else:
#                     x_l = 1/L*(t3.tt_dense_matmul(weight_t,x_t).transpose(0,1)+self.bias) + S*torch.ones(self.out_features)
#                 #print('x_l shape:', x_l.shape)
#             for i in range(1,self.iter_num):
#                 if torch.norm(x_l) > self.epsilon:
#                     x_l = x_l - 1/L*t3.tt_dense_matmul(t3.tt_tt_mul(weight_t,self.weight),x_l.t()).transpose(0,1) + 1/L*t3.tt_dense_matmul(weight_t, x_t).transpose(0, 1) + self.bias
#                 else:
#                     #print('activated!!!')
#                     x_l = x_l - 1/L*t3.tt_dense_matmul(t3.tt_tt_mul(weight_t,self.weight),x_l.t()).transpose(0,1) + 1/L*t3.tt_dense_matmul(weight_t, x_t).transpose(0, 1) + self.bias + S*torch.ones(self.out_features)
#             return x_l
