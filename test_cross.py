import torch
import tntorch as tn
import numpy as np

#test TT-cross approximation to compute ReLU
#manipute element of tensor but still out of curse of dimensionality
device = torch.device("cuda:0")
full = torch.randn(18,18)
print('full shape', full.shape)

X, Y, Z = np.meshgrid(range(128), range(128), range(128))
full = torch.Tensor(np.sqrt(np.sqrt(X)*(Y+Z) + Y*Z**2)*(X + np.sin(Y)*np.cos(Z)))  # Some analytical 3D function
print(full.shape)

def relu(x):
    if x < 0:
        return 0
    else:
        return x


relu_t = tn.cross(function=relu, tensors=full)
print('relu_t',relu_t)

truth_relu = torch.ones(128,128,128)
for i in range(128):
    for j in range(128):
        for k in range(128):
            #print('element',full[i,j,k])
            if full[i,j,k] < 0.0:
                truth_relu[i,j,k] = 0
            else:
                truth_relu[i,j,k] = full[i,j,k]

print('truth ReLU:', truth_relu)
truth_t = tn.Tensor(truth_relu,ranks_tt=4)
print(tn.relative_error(truth_t,relu_t))
