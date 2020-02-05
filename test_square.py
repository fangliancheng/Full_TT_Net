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
N, D_in, H, D_out = 4,4,4,4

# Create random Tensors to hold input and outputs.
x = torch.randn(N, D_in, device=device, dtype=dtype)
#construct orthogonal matrix
q,r = torch.qr(x)
x = torch.diag(torch.Tensor([0.75,1,2,3])).to(device)
print("x:", x)

y = torch.randn(N, D_out, device=device, dtype=dtype)

# Create random Tensors for weights.
#w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w1 = torch.eye(4, dtype=dtype, device=device, requires_grad=True)
#w1 = q.clone().detach().requires_grad_(True).to(device)
#w1 = diag_weight.clone().detach().requires_grad_(True).to(device)
print('w1:', w1)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

# #Method 2
#         eig_dict_xtx = {}
#         n_eigens_v = curr_core.shape[1]
#         print('curr_core shape:', curr_core.shape)
#         xtx = curr_core.t().mm(curr_core)
#         ei_value, _ = torch.eig(xtx)
#         #print('eigen-value:', ei_value)
#
#         # For symmetric pd matrix, SVD and eigen-decomp coincide, columns of u_xxt is eigen-vector of xxt, which is also left singular vector of x
#         # columns of v_xtx is eigen-vector of xtx, which is also right singular vector of x
#         with torch.no_grad():
#             _, _, v_xtx = torch.svd(xtx)
#             for i in range(n_eigens_v):
#                 eig_dict_xtx.update({str(i): v_xtx[:, i][..., None]})
#
#         power_layer = power_iteration_once.apply
#         for i in range(n_eigens_v):
#             # columns of V
#             eig_dict_xtx[str(i)] = power_layer(xtx, eig_dict_xtx[str(i)])
#         V_column = []
#         for i in eig_dict_xtx.keys():
#             V_column.append(eig_dict_xtx[i])
#         V = torch.cat(V_column, dim=1)
#         #we solve U s.t UV' = curr_core, take transpose, VU' = curr_core'
#         U_t, _ = torch.solve(curr_core.permute(1, 0), V)
#         U = U_t.t()


learning_rate = 1e-6
for t in range(500):

    #ip1 have eigen value 1 10 50 100
    ip1 = (w1.mm(x)).mm(w1.t()).contiguous().view(4, -1)
    eig_dict_xtx = {}
    n_eigens_v = ip1.shape[1]
    print("ip1 shape:", ip1.shape)

    #make sure its square matrix
    assert(ip1.shape[0] == ip1.shape[1])
    """ip1 = w1*x*w1^T, ip1^T*ip1 = w1*x*w1^T*w1*x*w1^T = w1*x*x*w1^T, so xtx will have eigenvalue/singular value [1,100,400,900], so is PD"""
    xtx = ip1.t().mm(ip1)
    print('xtx:', xtx)
    #pdb.set_trace()

    with torch.no_grad():
        #pdb.set_trace()
        print("begin svd")
        print("cond number for input of svd:", LA.cond(xtx.cpu().data.numpy()))
        _, _, v_xtx = torch.svd(xtx)
        for i in range(n_eigens_v):
            eig_dict_xtx.update({str(i): v_xtx[:, i][..., None]})

    power_layer = power_iteration_once.apply
    for i in range(n_eigens_v):
        # columns of V
        eig_dict_xtx[str(i)] = power_layer(xtx, eig_dict_xtx[str(i)])
    V_column = []
    for i in eig_dict_xtx.keys():
        V_column.append(eig_dict_xtx[i])
    V = torch.cat(V_column, dim=1)

    #U_t, _ = torch.solve(ip1.permute(1, 0), V)
    #print('torch.solve completed!')
    #U = U_t.t()

    y_pred = V.mm(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    # Use autograd to compute the backward pass.
    print("begin backward!")
    loss.backward()

    # Update weights using gradient descent
    print("begin update!")
    with torch.no_grad():
        print('w1,2 grad:', w1.grad, w2.grad)
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # Manually zero the gradients after updating weights
        w1.grad.zero_()
        w2.grad.zero_()
