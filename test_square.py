import numpy as np
import torch
from t3nsor.utils import svd_fix
from eigen_backward import *
import pdb
from numpy import linalg as LA

dtype = torch.float
device = torch.device("cuda")

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 4, 4, 4, 4

# Create random Tensors to hold input and outputs, x is made to be diagonal matrix
diag_x = torch.diag(torch.Tensor([0.75, 1, 2, 3])).to(device)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Create random Tensors for weights. w1 is initialize as identity matrix
w1 = torch.eye(4, dtype=dtype, device=device, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):

    """ip1=w1*x*w1^T have eigen value 0.75,1,2,3"""
    ip1 = (w1.mm(diag_x)).mm(w1.t())

    """xtx = ip1^T*ip1, xtx initially have eigenvalue/singular value [0.75^2,1,4,9]"""
    xtx = ip1.t().mm(ip1)
    print('xtx:', xtx)

    eig_dict_xtx = {}
    n_eigens_v = ip1.shape[1]
    with torch.no_grad():
        print("cond number for input of svd:", LA.cond(xtx.cpu().data.numpy()))
        _, _, v_xtx = torch.svd(xtx)
        for i in range(n_eigens_v):
            eig_dict_xtx.update({str(i): v_xtx[:, i][..., None]})

    power_layer = power_iteration_once.apply
    for i in range(n_eigens_v):
        # columns of V
        eig_dict_xtx[str(i)] = power_layer(xtx, eig_dict_xtx[str(i)])

    #Use eigen-vector as columns of V
    V_column = []
    for i in eig_dict_xtx.keys():
        V_column.append(eig_dict_xtx[i])
    V = torch.cat(V_column, dim=1)

    y_pred = V.mm(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).mean()
    print('loss:', loss)
    if t % 100 == 99:
        print(t, loss.item())

    # Use autograd to compute the backward pass.
    loss.backward()

    # Update weights using gradient descent
    with torch.no_grad():
        print('w1,2 grad:', w1.grad, w2.grad)
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # Manually zero the gradients after updating weights
        w1.grad.zero_()
        w2.grad.zero_()
