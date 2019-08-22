import torch
import t3nsor as t3
import time
import numpy as np
from t3nsor import TensorTrain
#input will be original grad
def rieman_proj_first_core(grad):
    ndims = grad.ndims
    #dtype = grad.dtype
    shape = grad.raw_shape
    tt_ranks = grad.ranks
    first_tt_core = grad.tt_cores[0]
    #normalization
    for i in range(shape[0]):
        slice = torch.unsqueeze(first_tt_core[:,i,:]/torch.norm(first_tt_core[:,i,:]),1)
        if i ==0:
            curr_core = slice
        else:
            curr_core = torch.cat((curr_core,slice),dim=1)
    #print('calling first core:', curr_core)
    return curr_core

def rieman_proj_last_core(grad):
    ndims = grad.ndims
    #dtype = grad.dtype
    shape = grad.raw_shape
    tt_ranks = grad.ranks
    last_tt_core = grad.tt_cores[-1]
    #normalization
    for i in range(shape[-1]):
        slice = torch.unsqueeze(last_tt_core[:,i,:]/torch.norm(last_tt_core[:,i,:]),1)
        if i ==0:
            curr_core = slice
        else:
            curr_core = torch.cat((curr_core,slice),dim=1)
    return curr_core


#input will be original grad
def skew_proj(grad):
    ndims = grad.ndims
    #dtype = grad.dtype
    #shape is 2 dimensional array
    shape = grad.raw_shape
    new_cores = []
    new_cores.append(rieman_proj_first_core(grad))
    #print(new_cores)
    tt_ranks = grad.ranks
    for core_index in range(1,ndims-1):
        for j in range(shape[core_index]):
            tt_core = grad.tt_cores[core_index]
            slice = tt_core[:,j,:]
            #unsqueeze in the mid dimension
            skew = torch.unsqueeze((slice - torch.t(slice))/2,1)
            if j == 0:
                curr_core = skew
            else:
                curr_core = torch.cat((curr_core,skew),dim=1)
        new_cores.append(curr_core)
    new_cores.append(rieman_proj_last_core(grad))
    #print(new_cores)
    print('skew project finished!')
    return TensorTrain(new_cores)


#for intermidiate TT cores, do cayley matplotlib
#for first and last TT cores, do riemannian update
def Cayley(ref):
    print('Cayley started!')
    ndims = ref.ndims
    #dtype = ref.dtype
    shape = ref.raw_shape
    tt_ranks = ref.ranks
    new_cores = []
    new_cores.append(ref.tt_cores[0])
    output = []
    #internal tt cores
    for core_index in range(1,ndims-1):
        #print('outter loop #',core_index)
        for j in range(shape[core_index]):
            #print('inner loop #',j)
            tt_core = ref.tt_cores[core_index]
            slice = tt_core[:,j,:]
            print('slice shape',slice.shape)
            identity = torch.eye(tt_ranks[core_index])
            print('identity shape',identity.shape)
            cayley = torch.unsqueeze(torch.matmul(identity - slice, torch.inverse(identity + slice)),dim=1)
            print('inverse computing finished!')
            if j ==0:
                curr_core = cayley
            else:
                curr_core = torch.cat((curr_core,cayley),dim=1)
        print('appending a new core...')
        new_cores.append(curr_core)
    new_cores.append(ref.tt_cores[-1])
    print('new cores', new_cores[0].shape)
    print('Cayley finished!')
    return TensorTrain(new_cores)

alpha=0.2
shape = 3*[10]
one_tt = t3.tensor_ones(shape)
print('one_tt',one_tt.ranks)
minus_one_tensor = -torch.ones(shape)
minus_one_tt = t3.to_tt_tensor(minus_one_tensor,max_tt_rank=1)
print('actual rank',minus_one_tt.ranks)

minus_alpha_tensor = -alpha*torch.ones(shape)
minus_alpha_tt = t3.to_tt_tensor(minus_alpha_tensor,max_tt_rank=1)
print('actual rank',minus_alpha_tt.ranks)
#print(minus_alpha_tt.full())

#print('1',one_tt.tt_cores())
input_dense = np.arange(-500,500).reshape(10,10,10).astype(np.float32)
init_x = t3.to_tt_tensor(torch.tensor(input_dense),max_tt_rank=2)

x_update = init_x
print('x_update:', x_update)
for i in range(5000):
    gradF = t3.multiply(x_update, t3.add(t3.multiply(x_update,x_update), minus_one_tt))
    gradF = t3.round_tt(gradF,4)
    riemannian_grad = skew_proj(gradF)
    print('riemannian grad ranks:', riemannian_grad.ranks)
    x_update = Cayley(t3.multiply(minus_alpha_tt,riemannian_grad))
    print('loss',torch.norm(0.5*(t3.multiply(x_update,x_update).full() - one_tt.full())))
