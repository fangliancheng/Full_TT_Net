import torch
import numpy as np
import torch.nn as nn
import t3nsor as t3
import math
from t3nsor.utils import cayley
from t3nsor import TensorTrainBatch
from t3nsor import TensorTrain
from t3nsor.utils import svd_fix
import pdb


# This is intended to be an anolog to standard FC layer
# Standard FC layer: W*x is equivalent to doing multiple times of inner product(weighted sum of input) between W and x
# Thus, when the input is a TensorTrain, we initilize weight as the same TT shape as input, and multiple times of inner product between tt_cores of W and x
class core_wise_linear(nn.Module):
    def __init__(self, shape, tt_rank, in_channels, out_channels, settings):
        super(core_wise_linear, self).__init__()
        # initilize weight as the same TT as input
        self.ndims = len(shape)
        self.num_parameter = int(shape[0] * tt_rank) + int(shape[-1] * tt_rank) + int(
            tt_rank * tt_rank * np.sum(shape[1:self.ndims - 1]))
        self.shape = shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_shape = shape
        # I = in_channels*self.num_parameter
        self.weight = nn.Parameter(torch.Tensor(in_channels * self.num_parameter, out_channels))
        self.batch_size = settings.BATCH_SIZE
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))
        # nn.init.uniform_(self.weight,a=-10,b=10)

    def forward(self, input):
        # print(self.weight[:,0])
        # input is 3 TensorTrainBatch:R,G,B
        # pdb.set_trace()
        list_in_list = [i.tt_cores for i in input]
        # [TensorTrain1.tt_cores,TensorTrain2.tt_cores,TensorTrain3.tt_cores]
        long_1 = len(list_in_list)
        # reconstruct input shape
        I = torch.Tensor(self.in_channels * self.num_parameter, self.batch_size).to(input[0].tt_cores[0].device)
        start = 0
        for tt_batch_iter in range(0, long_1):
            long_2 = len(list_in_list[tt_batch_iter])
            for core_idx in range(0, long_2):
                temp = list_in_list[tt_batch_iter][core_idx].view(-1, self.batch_size)
                I[start:start + temp.shape[0], :] = temp
                start = start + temp.shape[0]
        return torch.einsum('io,ib->bo', self.weight, I)


class tt_to_dense(nn.Module):
    def __init__(self):
        super(tt_to_dense, self).__init__()

    def forward(self, input):
        temp = t3.tt_to_dense(input)
        return temp


#used in pre_FTT_net
class layer_to_tt_matrix(nn.Module):
    def __init__(self, settings):
        super(layer_to_tt_matrix, self).__init__()
        self.batch_size = settings.BATCH_SIZE
        self.tt_shape = settings.TT_MATRIX_SHAPE
        self.tt_rank = settings.TT_RANK
        self.settings = settings

    def forward(self, input):
        # set channel to be 1
        input = input.view(self.batch_size, 1, 512)
        # pdb.set_trace()
        # convert input from dense format to TT format, specificaly, TensorTrainBatch
        return t3.input_to_tt_matrix(input, self.settings)[0]

#
# class layer_to_tt_tensor(nn.Module):
#     def __init__(self, settings):
#         super(layer_to_tt_tensor, self).__init__()
#         self.batch_size = settings.BATCH_SIZE
#         self.tt_shape = settings.TT_SHAPE
#         self.tt_rank = settings.TT_RANK
#         self.settings = settings
#
#     def forward(self, input):
#         # set channel to be 1
#         input = input.view(self.batch_size, 1, 8, 64)
#         # pdb.set_trace()
#         # convert input from dense format to TT format, specificaly, TensorTrainBatch
#         return t3.input_to_tt_tensor(input, self.settings)


#conterpart layer: layer_to_tt_tensor, where we convert raw data to TT by a deterministic alg TT-SVD
#In this layer we add matrix multiplication to output of internal SVD in TT-SVD(a sequence of SVD and reshape)
class layer_to_tt_tensor_learnable(nn.Module):
    def __init__(self, shape, tt_rank, settings):
        super(layer_to_tt_tensor_learnable, self).__init__()
        self.settings = settings
        self.ndims = len(shape)
        self.picture_shape = [64, 3, 32, 32]
        self.batch_size = settings.BATCH_SIZE

        self.inside_svd_weights_s_shape = [self.batch_size, 3, self.ndims-1, self.settings.TT_RANK]
        #self.inside_svd_weights_u_v_shape = [self.batch_size, 3, self.ndims-1, self.settings.TT_RANK, self.settings.TT_RANK]
        self.inside_svd_weights_u_v_shape = [3, self.ndims-1, self.settings.TT_RANK, self.settings.TT_RANK]

        #self.weight = nn.Parameter(torch.Tensor(*self.picture_shape))
        #the following weight is for S component in output of SVD
        self.weight_s_hadmard = nn.Parameter(torch.Tensor(*self.inside_svd_weights_s_shape))
        #the following weight is for U,V component in output of SVD
        self.weight_u_mm = nn.Parameter(torch.Tensor(*self.inside_svd_weights_u_v_shape))
        self.weight_v_mm = nn.Parameter(torch.Tensor(*self.inside_svd_weights_u_v_shape))
        self.max_tt_rank = settings.TT_RANK
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_normal_(self.weight_s_hadmard, a=math.sqrt(5))
        nn.init.kaiming_normal_(self.weight_u_mm, a=math.sqrt(5))
        nn.init.kaiming_normal_(self.weight_v_mm, a=math.sqrt(5))
        # nn.init.uniform_(self.weight,a=-10,b=10)

    def forward(self, input):
        num_channels = input.shape[1]
        input_tt = []
        for num_c in range(num_channels):
            tt_cores_curr = []
            tt_batch_cores_curr = []

            for batch_iter in range(self.batch_size):
                # tt_cores_curr += t3.to_tt_tensor(input[batch_iter,num_c,:,:].view(settings.TT_SHAPE),max_tt_rank=settings.TT_RANK).tt_cores
                if self.settings.LEARNABLE:
                    # hadmard product
                    # tens = torch.mul(input[batch_iter,num_c,:,:],self.weight[batch_iter,num_c,:,:]).view(self.settings.TT_SHAPE)
                    tens = input[batch_iter, num_c, :, :].view(self.settings.TT_SHAPE)
                shape = tens.shape
                d = len(shape)
                max_tt_rank = np.array(self.max_tt_rank).astype(np.int32)
                if max_tt_rank.size == 1:
                    max_tt_rank = [int(self.max_tt_rank), ] * (d + 1)

                ranks = [1] * (d + 1)
                tt_cores = []

                for core_idx in range(d - 1):
                    curr_mode = shape[core_idx]
                    rows = ranks[core_idx] * curr_mode

                    tens = tens.view(rows, -1)
                    columns = tens.shape[1]
                    u, s, v = svd_fix(tens)

                    if max_tt_rank[core_idx + 1] == 1:
                        ranks[core_idx + 1] = 1
                    else:
                        #print(min(max_tt_rank[core_idx + 1], rows, columns))
                        ranks[core_idx + 1] = min(max_tt_rank[core_idx + 1], rows, columns)

                    uu = u[:, 0:ranks[core_idx + 1]]

                    #perform matrix multiplication
                    if core_idx >= 1:
                        #uu = torch.mm(uu, self.weight_u_mm[batch_iter, num_c, core_idx, :, :])
                        uu = torch.mm(uu, self.weight_u_mm[num_c, core_idx, :, :])
                    #pdb.set_trace()
                    #ss = s[0:ranks[core_idx + 1]]
                    assert(self.weight_s_hadmard[batch_iter, num_c, core_idx, :].shape[0] == s[0:ranks[core_idx+1]].shape[0])
                    ss = s[0:ranks[core_idx+1]]
                    if core_idx >= 1:
                        ss = torch.mul(self.weight_s_hadmard[batch_iter, num_c, core_idx, :], ss)
                    vv = v[:, 0:ranks[core_idx + 1]]
                    if core_idx >= 1:
                        #vv = torch.mm(vv, self.weight_v_mm[batch_iter, num_c, core_idx, :, :])
                        vv = torch.mm(vv, self.weight_v_mm[num_c, core_idx, :, :])
                    #pdb.set_trace()
                    core_shape = (ranks[core_idx], curr_mode, ranks[core_idx + 1])
                    tt_cores.append(uu.view(*core_shape))
                    tens = torch.matmul(torch.diag(ss), vv.permute(1, 0))
                last_mode = shape[-1]

                core_shape = (ranks[d - 1], last_mode, ranks[d])
                tt_cores.append(tens.view(core_shape))

                tt_cores_curr += tt_cores
            tt_core_curr_unsq = [torch.unsqueeze(i, dim=0) for i in tt_cores_curr]
            assert(len(self.settings.TT_SHAPE) == 4)
            for shift in range(1, 5):
                tt_batch_cores_curr.append(
                    torch.cat(tt_core_curr_unsq[shift - 1:(self.settings.BATCH_SIZE - 1) * 4 + shift:4], dim=0))
            input_tt.append(TensorTrainBatch(tt_batch_cores_curr))
        assert (len(input_tt) == 3)
        return input_tt


# input: TensorTrainBatch1
class TTFC(nn.Module):
    def __init__(self, shape, tt_rank=4, in_channels=120, init=None):
        super(TTFC, self).__init__()
        # example: shape = [7,8,9] d=3 n1=7 n2=8 n3=9
        self.ndims = len(shape)
        self.jj = int(np.sum(shape))
        self.shape = shape
        self.in_channels = in_channels
        self.init_shape = [8, 8, 8, 8]
        # self.init_shape = [4,5,4,5]
        if init is None:
            # init = t3.glorot_initializer(self.init_shape, tt_rank=tt_rank)
            # init = t3.tensor_with_random_cores(self.init_shape,tt_rank = tt_rank)
            epsilon = 0
            init = t3.tensor_with_random_cores_epsilon(self.init_shape, tt_rank=tt_rank, epsilon=epsilon)
        self.weight = init.to_parameter()
        self.parameters = self.weight.parameter

    def forward(self, input):
        # print('weight shape:',self.weight)
        # list length: in_channels, each element in list will contribute to one column of the output [batch_size, in_channels]
        list_in_list = [i.tt_cores for i in input]

        # stacked_core_list = []
        # for core_iter in range(0,self.ndims):
        #     temp_list = []
        #     for i in range(0,in_channels):
        #         temp_list += re_organized[i + core_iter*in_channels]
        #     stacked_core_list += torch.stack(*temp_list,dim=0)
        norm_matrix = 0
        for tt_batch_iter in range(0, self.in_channels):
            curr_batch_cores = list_in_list[tt_batch_iter]
            # curr_batch_cores is a list of tt_cores, each tt_core is 4 dimension: batch_size, r_1, n_1, r_2
            count_2_norm = 0
            for core_iter in range(0, len(curr_batch_cores) - 1):

                # weight in exactly same TT format as input
                input_curr_core = curr_batch_cores[core_iter]
                weight_curr_core = self.weight.tt_cores[core_iter]
                n = input_curr_core.shape[2]
                for slice_iter in range(0, n):
                    # out.shape: batch_size, r, r
                    out = torch.einsum('jk,bji->bki', weight_curr_core[:, slice_iter, :],
                                       input_curr_core[:, :, slice_iter, :])
                    out_t = 1 / 2 * (out.transpose(1, 2) - out)

                    # print(out_t.shape)
                    # pdb.set_trace()

                    # print('out shape:',out.shape)
                    # if max(torch.norm(out,dim=(1,2)))<1e-8:
                    #     print('zero value in norm!!!!!!')
                    # gradient of zero norm is nan, need to add an eps
                    eps = 1e-5
                    count_2_norm = count_2_norm + torch.norm(out_t + eps * torch.ones(out.shape).to(out.device),
                                                             dim=(1, 2)) ** 2
            # norm_list will be a list of torch tensors
            # print('norm_matrix shape222:',norm_matrix)
            count_2_norm_sqrt = torch.sqrt(count_2_norm + eps)
            if tt_batch_iter == 0:
                # print('count_2_norm shape:',count_2_norm.shape)
                norm_matrix = torch.unsqueeze(count_2_norm_sqrt, dim=1)
            else:
                # pdb.set_trace()
                norm_matrix = torch.cat((norm_matrix, torch.unsqueeze(count_2_norm_sqrt, dim=1)), dim=1)


        # print(norm_matrix.view(out.shape[0],-1))
        # pdb.set_trace()
        return norm_matrix.view(out.shape[0], -1)
        # shape of output: [batch_size,in_channels]


class TTConv(nn.Module):
    # N is in_channels
    # in_channels=3 means input will be 3 TensorTrainBatch, out_channels=84 means output will be 84 TensorTrainBatch
    def __init__(self, shape, in_channels, out_channels):
        super(TTConv, self).__init__()
        # example: shape = [4,8,4,8] d=4 n1=4 n2=8 n3=4, n4=8
        self.ndims = len(shape)
        self.jj = int(np.sum(shape))
        # initilize the same as nn.Linear
        self.weight = nn.Parameter(torch.Tensor(in_channels, self.jj, out_channels))
        self.shape = shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data = nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # normalize weight
        for i in range(0, self.jj):
            for j in range(0, self.out_channels):
                self.weight.data[:, i, j] = self.weight.data[:, i, j] / self.weight.data[:, i, j].sum()
        epsilon = 0
        self.weight.data.add_(epsilon)
        # print(self.weight.data.shape)
        # print('sum:',self.weight.data[:,0,0].sum())
        # self.weight.data.constant_(1/self.N)
        # nn.init.constant_(self.weight, 1/self.in_channels)

    # input:list(TensorTrainBatch1,TensorTrainBatch2,TensorTrainBatch3)
    def forward(self, input):
        list_in_list = [i.tt_cores for i in input]
        re_organized = []
        for core_iter in range(0, self.ndims):
            temp = [i[core_iter] for i in list_in_list]
            re_organized += temp
        # print('re_organized length:',len(re_organized))
        # re_organized will be a list: [TenosorTrainBatch_1.cores[0],TensorTrainBatch_2.cores[0],...,TensorTrainBatch_N.cores[0],
        #                              TensorTrainBatch_1.cores[1], TensorTrainBatch_2.cores[1],...,TensorTrainBatch_N.cores[1],
        #                               ......
        #
        # cores: (batch_size, r_1, slice_iter, r_2)
        # Use TensorTrainBatch_1.cores[0], TensorTrainBatch_2.cores[0],... to construct output_TensorTrainBatch.cores[0]
        # stack these cores to becomes a big tensor: [in_channels, batch_size, r_1, slice_iter, r_2]
        stacked_core_list = []
        for core_iter in range(0, self.ndims):
            temp_list = []
            for i in range(0, self.in_channels):
                temp_list.append(re_organized[i + core_iter * self.in_channels])
            stacked_core_list.append(torch.stack(temp_list, dim=0))
        # stacked_core_list will be [stacked_1,stacked_2,stracked_3,stracked_4]

        # apply logm
        # Replace this by inverse cayley
        # for i in range(1,self.ndims-1):
        #     stacked_core_list[i] = 1/2*(stacked_core_list[i].transpose(2,4) - stacked_core_list[i])
        #     #pdb.set_trace()
        for core_iter in range(1, self.ndims - 1):
            for slice_iter in range(0, self.shape[core_iter]):
                stacked_core_list[core_iter][:, :, :, slice_iter, :] = cayley(
                    stacked_core_list[core_iter][:, :, :, slice_iter, :])

        out_slice_list = []
        cat_slice = []
        slice_cumulative_count = 0
        for core_iter in range(0, self.ndims):
            if core_iter == 0:
                for slice_iter in range(0, self.shape[core_iter]):
                    out_slice_list.append(torch.unsqueeze(
                        torch.einsum('io,ibkj->obkj', self.weight[:, slice_cumulative_count, :],
                                     stacked_core_list[core_iter][:, :, :, slice_iter, :]), dim=3))
                    slice_cumulative_count += 1
                cat_slice.append(torch.cat(out_slice_list, dim=3))
                out_slice_list = []
            elif core_iter == self.ndims - 1:
                for slice_iter in range(0, self.shape[core_iter]):
                    out_slice_list.append(torch.unsqueeze(
                        torch.einsum('io,ibkj->obkj', self.weight[:, slice_cumulative_count, :],
                                     stacked_core_list[core_iter][:, :, :, slice_iter, :]), dim=3))
                    slice_cumulative_count += 1
                cat_slice.append(torch.cat(out_slice_list, dim=3))
                out_slice_list = []
            else:
                # pdb.set_trace()
                for slice_iter in range(0, self.shape[core_iter]):
                    out_slice_list.append(torch.unsqueeze(cayley(
                        torch.einsum('io,ibkj->obkj', self.weight[:, slice_cumulative_count, :],
                                     stacked_core_list[core_iter][:, :, :, slice_iter, :])), dim=3))
                    slice_cumulative_count += 1
                # cat_slice already duplicated!!!
                cat_slice.append(torch.cat(out_slice_list, dim=3))
                out_slice_list = []

        # shape of element in cat_slice: [out_channels,batch_size,r_1,slice_iter,r_2]
        # split along the out_channels dimension
        # splited_list is a list of tuple
        splited_list = [torch.split(curr_core, split_size_or_sections=1, dim=0) for curr_core in cat_slice]
        # pdb.set_trace()
        re_organized_splited_list = []
        for iter in range(0, self.out_channels):
            temp = [torch.squeeze(i[iter], 0) for i in splited_list]
            re_organized_splited_list += temp
        ttbatch_list = []
        for i in range(0, self.out_channels):
            ttbatch_list.append(
                TensorTrainBatch(re_organized_splited_list[i * self.ndims:i * self.ndims + self.ndims:1],
                                 convert_to_tensors=False))
        # pdb.set_trace()
        return ttbatch_list


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
        # self.weight_t = t3.transpose(self.weight,convert_to_tensors=False)

        if bias:
            self.bias = torch.nn.Parameter(1e-2 * torch.ones(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # Hint: put all forward pass computation in forward.
        # Wrong thing: if we put t3.transpose computation in init, model.cuda() cannot
        # transform output of transpose to cuda because it is not claimed to be nn.Parameter
        # weight_t = self.weight_t
        weight_t = t3.transpose(self.weight, convert_to_tensors=False)
        # print('weight_t device',weight_t.tt_cores[0].device)
        # print('weight device',self.weight.tt_cores[0].device)
        x_t = x.transpose(0, 1)
        if self.bias is None:
            temp = t3.tt_dense_matmul(weight_t, x_t).transpose(0, 1)
            return temp.to(x.device)
        else:
            temp = t3.tt_dense_matmul(weight_t, x_t).transpose(0, 1) + self.bias
            # print('x device:',x.device)
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
        weight_t = t3.transpose(self.weight, False)
        # print('wight_t shape:', weight_t.shape)
        x_t = x.transpose(0, 1)
        # print('x_t shape:',x_t.shape)

        if self.bias is None:
            return t3.tt_dense_matmul(weight_t, x_t).transpose(0, 1)
        else:
            L = self.L
            S = self.S
            if self.iter_num == 1:
                if torch.norm(x) > self.epsilon:
                    return 1 / L * (t3.tt_dense_matmul(weight_t, x_t).transpose(0, 1) + self.bias)
                else:
                    return 1 / L * (t3.tt_dense_matmul(weight_t, x_t).transpose(0, 1) + self.bias) + S * torch.ones(self.out_features)
            else:
                # iter_num > 1
                if torch.norm(x) > self.epsilon:
                    x_l = 1 / L * (t3.tt_dense_matmul(weight_t, x_t).transpose(0, 1) + self.bias)
                else:
                    x_l = 1 / L * (t3.tt_dense_matmul(weight_t, x_t).transpose(0, 1) + self.bias) + S * torch.ones(self.out_features).to(x_t.device)
                # print('x_l shape:', x_l.shape)
            for i in range(1, self.iter_num):
                if torch.norm(x_l) > self.epsilon:
                    x_l = x_l - 1 / L * t3.tt_dense_matmul(t3.tt_tt_matmul(weight_t, self.weight), x_l.t()).transpose(0,
                                                                                                                      1) + 1 / L * t3.tt_dense_matmul(
                        weight_t, x_t).transpose(0, 1) + self.bias
                else:
                    print('activated!!!')
                    x_l = x_l - 1 / L * t3.tt_dense_matmul(t3.tt_tt_matmul(weight_t, self.weight), x_l.t()).transpose(0,
                                                                                                                      1) + 1 / L * t3.tt_dense_matmul(
                        weight_t, x_t).transpose(0, 1) + self.bias + S * torch.ones(self.out_features).to(x_t.device)
            return x_l


class FTTLinear(nn.Module):
    def __init__(self, in_features=None, out_features=None, bias=False, init=None, shape=[[3, 4, 8, 4, 8], [1, 2, 5, 1, 1]],
                 auto_shapes=False, d=3, tt_rank=3, stddev=None, auto_shape_mode='ascending',
                 auto_shape_criterion='entropy',
                 ):

        super(FTTLinear, self).__init__()

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
        # put all forward computation in forward method, for .cuda()
        # self.weight_t = t3.transpose(self.weight,convert_to_tensors=False)

        bias_shape = init.raw_shape
        if bias:
            self.bias = torch.nn.Parameter(1e-2 * torch.ones(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # x will be a TensorTrainBatch
        # Hint: put all forward pass computation in forward.
        # Wrong thing: if we put t3.transpose computation in init, model.cuda() cannot
        # transform output of transpose to cuda because it is not claimed to be nn.Parameter
        # weight_t = self.weight_t
        # weight_t = t3.transpose(self.weight,convert_to_tensors=False)
        # x_t = x.transpose(0, 1)
        if self.bias is None:
            temp = t3.tt_tt_matmul(x, self.weight)
            # pdb.set_trace()
            return temp
        else:
            # pdb.set_trace()
            # TODO: initilize self.bias in TT format
            temp = t3.add(t3.tt_tt_matmul(x, self.weight), self.bias)
            return temp


class FTT_Solver(nn.Module):
    def __init__(self,
                in_features=None,
                out_features=None,
                bias=True,
                init=None,
                shape=None,
                auto_shapes=False,
                d=3,
                tt_rank=8,
                iter_num=1,
                l=1,
                s=-1,
                epsilon=1,
                auto_shape_mode='ascending',
                auto_shape_criterion='entropy'):

        super(FTT_Solver, self).__init__()

        if auto_shapes:
            print('auto_shape working...')
            if in_features is None or out_features is None:
                raise ValueError("Shape is not specified")

            in_quantization = t3.utils.auto_shape(in_features, d=d, criterion=auto_shape_criterion, mode=auto_shape_mode)
            out_quantization = t3.utils.auto_shape(out_features, d=d, criterion=auto_shape_criterion, mode=auto_shape_mode)

            shape = [in_quantization, out_quantization]

        if init is None:
            if shape is None:
                raise ValueError(
                        "if init is not provided, please specify shape, or set auto_shapes=True")
        else:
            shape = init.raw_shape

        if init is None:
            init = t3.glorot_initializer(shape, tt_rank=tt_rank)
            init = t3.scalar_tt_mul(init, 1/5)

        self.init_tt_rank = tt_rank
        self.shape = shape
        #self.weight = init.to_parameter()
        self.parameter_weight = None
        self.parameter_bias = None
        self.iter_num = iter_num
        self.L = l
        self.S = s
        self.epsilon = epsilon
        self.out_features = out_features
        self.svd_matrix_round = None
        self.svd_matrix_orth = None

        if bias:
            #self.bias = torch.nn.Parameter(1e-2 * torch.ones(out_features))
            init_bias = t3.glorot_initializer(shape=[None, self.shape[1]])
            init_bias = t3.scalar_tt_mul(init_bias, 1/5)
            #self.test_weight = nn.Parameter(init_bias.tt_cores[0])
            #self.bias = init_bias.to_parameter()
            #self.bias = self.bias_to_parameter(init_bias)
            #manully register bias parameter:
            #self.register_parameter('bias', self.bias)
            self.weight, self.bias = self.weight_bias_to_parameter(init, init_bias)

        else:
            raise NotImplementedError
            #self.register_parameter('bias', None)

    def weight_bias_to_parameter(self, init_weight, init_bias):
        weight_list = nn.ParameterList([])
        bias_list = nn.ParameterList([])

        for weight_core in init_weight.tt_cores:
            """Can only append nn.Parameter to nn.ParameterList"""
            weight_core = nn.Parameter(weight_core)
            weight_list.append(weight_core)

        """ParameterList have only one memory, when it is changed, anything directly depend on it will be automatically updated"""
        #for test
        # para_data_list = [e.data for e in weight_bias_list]
        # weight_tt_data = TensorTrain(para_data_list, convert_to_tensors=False)
        # pdb.set_trace()

        weight_tt = TensorTrain(weight_list, convert_to_tensors=False)
        #pdb.set_trace()

        for bias_core in init_bias.tt_cores:
            """Can only append nn.Parameter to nn.ParameterList"""
            bias_core = nn.Parameter(bias_core)
            #core.is_tt = True
            bias_list.append(bias_core)

        """We must add this assignment to add the parameter to model"""
        """1.Construct nn.Parameterlist 2.Move weights to nn.Parameters 3.Append them to ParameterList 4.Assign Parameterlist to something"""
        self.parameter_weight = weight_list
        self.parameter_bias = bias_list

        bias_tt = TensorTrain(bias_list, convert_to_tensors=False)
        #tt_p = TensorTrain(new_bias_cores, convert_to_tensors=False)
        #test
        #core1 = tt_p.tt_cores[0]
        #tt_p._parameter = nn.ParameterList(tt_p.tt_cores)
        #print('core1:',core1)
        #tt_p._parameter = nn.ParameterList(core1)
        bias_tt._is_parameter = True
        weight_tt._is_parameter = True
        return weight_tt, bias_tt

    def forward(self, x):

        if self.bias is None:
            raise NotImplementedError
        else:
            L = self.L
            S = self.S
            if self.iter_num == 1:
                f_norm = t3.frobenius_norm_squared(x)
                batch_size = len(f_norm)
                tt_list = []
                for idx in range(batch_size):
                    if f_norm[idx] > self.epsilon:

                        output_tt = t3.scalar_tt_mul(t3.add(t3.tt_tt_matmul(t3.utils.get_element_from_batch(x, idx), self.weight), self.bias), 1/L)
                        rounded_output_tt, self.svd_matrix_round, self.svd_matrix_orth = t3.round(output_tt, self.init_tt_rank)
                        tt_list.append(rounded_output_tt)
                    else:
                        output_tt = t3.add(t3.scalar_tt_mul(t3.add(t3.tt_tt_matmul(t3.utils.get_element_from_batch(x, idx), self.weight), self.bias), 1/L), t3.scalar_tt_mul(t3.matrix_ones([None, self.shape[1]]), S))
                        rounded_output_tt, self.svd_matrix_round, self.svd_matrix_orth = t3.round(output_tt, self.init_tt_rank)
                        tt_list.append(rounded_output_tt)

                return t3.utils.tt_batch_from_list_of_tt(tt_list), self.svd_matrix_round, self.svd_matrix_orth
            else:
                raise NotImplementedError
            # # iter_num > 1
            #     pdb.set_trace()
            #     if t3.frobenius_norm_squared(x) > self.epsilon:
            #         x_l = 1/L*(t3.tt_tt_matmul(x, self.weight) + self.bias)
            #     else:
            #         x_l = 1/L*(t3.tt_tt_matmul(x, self.weight) + self.bias) + S*torch.ones(self.out_features)
            # for i in range(1, self.iter_num):
            #     if torch.norm(x_l) > self.epsilon:
            #         x_l = x_l - 1/L*t3.tt_tt_matmul(t3.tt_tt_matmul(self.weight, self.weight), x_l) + 1/L*t3.tt_tt_matmul(x, self.weight) + self.bias
            #     else:
            #         #add a random TT matrix
            #         x_l = x_l - 1/L*t3.tt_tt_matmul(t3.tt_tt_matmul(self.weight, self.weight), x_l) + 1/L*t3.tt_tt_matmul(x, self.weight) + self.bias + S*torch.ones(self.out_features)
            # return x_l
