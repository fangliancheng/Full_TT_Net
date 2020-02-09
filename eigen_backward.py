import torch as th
import torch.nn as nn
from time import sleep
import time

class power_iteration_once(th.autograd.Function):
    @staticmethod
    def forward(ctx, M, v_k, num_iter=19):
        '''
        :param ctx: used to save meterials for backward.
        :param M: n by n matrix.
        :param v_k: initial guess of leading vector.
        :return: v_k1 leading vector.
        '''
        ctx.num_iter = num_iter
        ctx.save_for_backward(M, v_k)
        return v_k

    @staticmethod
    def backward(ctx, grad_output):
        print('backward calling...')
        M, v_k = ctx.saved_tensors
        dL_dvk = grad_output
        I = th.eye(M.shape[-1], out=th.empty_like(M))
        numerator = I - v_k.mm(th.t(v_k))
        denominator = th.norm(M.mm(v_k)).clamp(min=1.e-5)
        ak = numerator / denominator
        term1 = ak
        q = M / denominator
        for i in range(1, ctx.num_iter + 1):
            ak = q.mm(ak)
            term1 += ak
        dL_dM = th.mm(term1.mm(dL_dvk), v_k.t())
        return dL_dM, ak