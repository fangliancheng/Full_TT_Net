from t3nsor import TensorTrainBatch
from t3nsor import TensorTrain
import t3nsor as t3
import torch
from t3nsor.utils import kronecker_product

def multiply(tt_left,tt_right):
    ndims = tt_left.ndims
    shape = tt_left.raw_shape
    #tt_ranks = shapes.lazy_tt_ranks(tt_left)
    new_cores = []
    for core_idx in range(ndims):
        for j in range(shape[core_idx]):
            left_curr_core = tt_left.tt_cores[core_idx]
            right_curr_core = tt_right.tt_cores[core_idx]
            left_slice = left_curr_core[:,j,:]
            right_slice = right_curr_core[:,j,:]
            #print('left_slice:',left_slice)
            #print('right slice', right_slice)
            out_slice = torch.unsqueeze(kronecker_product(left_slice,right_slice),dim=1)
            #print('out_slice',out_slice)
            if j == 0:
                out_core = out_slice
            else:
                out_core = torch.cat((out_core,out_slice),dim=1)
        #print('out_core！！！！！',out_core)
        new_cores.append(out_core)
    return TensorTrain(new_cores)


def add(tt_a, tt_b):
  """Internal function to be called from add for two TT-tensors.
  Does the actual assembling of the TT-cores to add two TT-tensors.
  """
  ndims = tt_a.ndims
  #dtype = tt_a.dtype
  shape = tt_a.raw_shape
  a_ranks = tt_a.ranks
  b_ranks = tt_b.ranks
  tt_cores = []
  for core_idx in range(ndims):
    a_core = tt_a.tt_cores[core_idx]
    b_core = tt_b.tt_cores[core_idx]
    if core_idx == 0:
      curr_core = torch.cat((a_core, b_core), dim=2)
    elif core_idx == ndims - 1:
      curr_core = torch.cat((a_core, b_core), dim=0)
    else:
      upper_zeros = torch.zeros((a_ranks[core_idx], shape[core_idx],
                              b_ranks[core_idx + 1]))
      lower_zeros = torch.zeros((b_ranks[core_idx], shape[core_idx],
                              a_ranks[core_idx + 1]))
      upper = torch.cat((a_core, upper_zeros), dim=2)
      lower = torch.cat((lower_zeros, b_core), dim=2)
      curr_core = torch.cat((upper, lower), dim=0)
    tt_cores.append(curr_core)
  return TensorTrain(tt_cores)


def gather_rows(tt_mat, inds):
    """
    inds -- list of indices of shape batch_size x d
    d = len(tt_mat.raw_shape[1])
    """
    cores = tt_mat.tt_cores
    slices = []
    batch_size = int(inds[0].shape[0])


    ranks = [int(tt_core.shape[0]) for tt_core in tt_mat.tt_cores] + [1, ]


    for k, core in enumerate(cores):
        i = inds[k]
        #core = core.permute(1, 0, 2, 3).to(inds.device)

        cur_slice = torch.index_select(core, 1, i)

        if k == 0:
            res = cur_slice

        else:
            res = res.view(batch_size, -1, ranks[k])
            curr_core = cur_slice.view(ranks[k], batch_size, -1)
            res = torch.einsum('oqb,bow->oqw', (res, curr_core))


    return res





        #slices.append(torch.index_select(core, 1, i).permute(1, 0, 2, 3))



    return TensorTrainBatch(slices, convert_to_tensors=False)


def transpose(tt_matrix):
    cores = []
    for core in tt_matrix.tt_cores:
        cores.append(core.transpose(1, 2))
    return TensorTrain(cores)

#Liancheng
def tt_tt_mul(tt_matrix_a, tt_matrix_b):
    """Multiplies two TT-matrices and returns the TT-matrix of the result.
  Args:
    tt_matrix_a: `TensorTrain` or `TensorTrainBatch` object containing
      a TT-matrix (a batch of TT-matrices) of size M x N
    tt_matrix_b: `TensorTrain` or `TensorTrainBatch` object containing
      a TT-matrix (a batch of TT-matrices) of size N x P
  Returns
    `TensorTrain` object containing a TT-matrix of size M x P if both arguments
      are `TensorTrain`s
    `TensorTrainBatch` if any of the arguments is a `TensorTrainBatch`
  Raises:
    ValueError is the arguments are not TT matrices or if their sizes are not
    appropriate for a matrix-by-matrix multiplication.
  """
  # # Both TensorTrain and TensorTrainBatch are inherited from TensorTrainBase.
  #   if not isinstance(tt_matrix_a, TensorTrainBase) or \
  #       not isinstance(tt_matrix_b, TensorTrainBase) or \
  #       not tt_matrix_a.is_tt_matrix() or \
  #       not tt_matrix_b.is_tt_matrix():
  #       raise ValueError('Arguments should be TT-matrices')
  #
  #   if not shapes.is_batch_broadcasting_possible(tt_matrix_a, tt_matrix_b):
  #       raise ValueError('The batch sizes are different and not 1, broadcasting is '
  #                    'not available.')

    ndims = tt_matrix_a.ndims
    if tt_matrix_b.ndims != ndims:
        raise ValueError('Arguments should have the same number of dimensions, '
                     'got %d and %d instead.' % (ndims, tt_matrix_b.ndims()))

    # Convert BatchSize 1 batch into TT object to simplify broadcasting.
    # tt_matrix_a = shapes.squeeze_batch_dim(tt_matrix_a)
    # tt_matrix_b = shapes.squeeze_batch_dim(tt_matrix_b)
    # is_a_batch = isinstance(tt_matrix_a, TensorTrainBatch)
    # is_b_batch = isinstance(tt_matrix_b, TensorTrainBatch)
    # is_res_batch = is_a_batch or is_b_batch
    # a_batch_str = 'o' if is_a_batch else ''
    # b_batch_str = 'o' if is_b_batch else ''
    # res_batch_str = 'o' if is_res_batch else ''
    # einsum_str = '{}aijb,{}cjkd->{}acikbd'.format(a_batch_str, b_batch_str,
    #                                             res_batch_str)
    a_batch_str = ''
    b_batch_str = ''
    res_batch_str = ''
    einsum_str = '{}aijb,{}cjkd->{}acikbd'.format(a_batch_str, b_batch_str,
                                                res_batch_str)
    result_cores = []
    a_shape = tt_matrix_a.raw_shape
    a_ranks = tt_matrix_a.ranks
    b_shape = tt_matrix_b.raw_shape
    b_ranks = tt_matrix_b.ranks

    for core_idx in range(ndims):
        a_core = tt_matrix_a.tt_cores[core_idx]
        b_core = tt_matrix_b.tt_cores[core_idx]
        curr_res_core = torch.einsum(einsum_str, a_core, b_core)

        res_left_rank = a_ranks[core_idx] * b_ranks[core_idx]
        res_right_rank = a_ranks[core_idx + 1] * b_ranks[core_idx + 1]
        left_mode = a_shape[0][core_idx]
        right_mode = b_shape[1][core_idx]

        core_shape = (res_left_rank, left_mode, right_mode,
                    res_right_rank)
        curr_res_core = torch.reshape(curr_res_core, core_shape)
        result_cores.append(curr_res_core)

    res_shape = (tt_matrix_a.raw_shape[0], tt_matrix_b.raw_shape[1])
    static_a_ranks = tt_matrix_a.ranks
    static_b_ranks = tt_matrix_b.ranks
    out_ranks = [a_r * b_r for a_r, b_r in zip(static_a_ranks, static_b_ranks)]

    return TensorTrain(result_cores)



def tt_dense_matmul(tt_matrix_a, matrix_b):
    """Multiplies a TT-matrix by a regular matrix, returns a regular matrix.
    Args:
    tt_matrix_a: `TensorTrain` object containing a TT-matrix of size M x N
    matrix_b: torch.Tensor of size N x P
    Returns
    torch.Tensor of size M x P
    """

    ndims = tt_matrix_a.ndims
    a_columns = tt_matrix_a.shape[1]
    b_rows = matrix_b.shape[0]
    if a_columns is not None and b_rows is not None:
        if a_columns != b_rows:
            raise ValueError('Arguments shapes should align got %d and %d instead.' %
                       (tt_matrix_a.shape, matrix_b.shape))

    a_shape = tt_matrix_a.shape
    a_raw_shape = tt_matrix_a.raw_shape
    b_shape = matrix_b.shape
    a_ranks = tt_matrix_a.ranks

    # If A is (i0, ..., id-1) x (j0, ..., jd-1) and B is (j0, ..., jd-1) x K,
    # data is (K, j0, ..., jd-2) x jd-1 x 1
    data = matrix_b.transpose(0, 1)
    #Liancheng +contiguous
    data = data.contiguous().view(-1, a_raw_shape[1][-1], 1)

    for core_idx in reversed(range(ndims)):
        curr_core = tt_matrix_a.tt_cores[core_idx]
        # On the k = core_idx iteration, after applying einsum the shape of data
        # becomes ik x (ik-1..., id-1, K, j0, ..., jk-1) x rank_k
        data = torch.einsum('aijb,rjb->ira', curr_core, data)
        if core_idx > 0:
          # After reshape the shape of data becomes
          # (ik, ..., id-1, K, j0, ..., jk-2) x jk-1 x rank_k
            new_data_shape = (-1, a_raw_shape[1][core_idx - 1], a_ranks[core_idx])
            data = data.contiguous().view(new_data_shape)

    # At the end the shape of the data is (i0, ..., id-1) x K
    return data.view(a_shape[0], b_shape[1])
