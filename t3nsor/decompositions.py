import numpy as np
import torch

from t3nsor.tensor_train import TensorTrain
from t3nsor.utils import svd_fix


def to_tt_tensor(tens, max_tt_rank=10, epsilon=None):

    shape = tens.shape
    d = len(shape)
    max_tt_rank = np.array(max_tt_rank).astype(np.int32)
    if max_tt_rank.size == 1:
        max_tt_rank = [int(max_tt_rank), ] * (d+1)

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
            ranks[core_idx + 1] = min(max_tt_rank[core_idx + 1], rows, columns)

        u = u[:, 0:ranks[core_idx + 1]]
        s = s[0:ranks[core_idx + 1]]
        v = v[:, 0:ranks[core_idx + 1]]
        core_shape = (ranks[core_idx], curr_mode, ranks[core_idx + 1])
        tt_cores.append(u.view(*core_shape))
        tens = torch.matmul(torch.diag(s), v.permute(1, 0))

    last_mode = shape[-1]

    core_shape = (ranks[d - 1], last_mode, ranks[d])
    tt_cores.append(tens.view(core_shape))

    return TensorTrain(tt_cores, convert_to_tensors=False)


def to_tt_matrix(mat, shape, max_tt_rank=10, epsilon=None):

    shape = list(shape)
    # In case shape represents a vector, e.g. [None, [2, 2, 2]]
    if shape[0] is None:
        shape[0] = np.ones(len(shape[1])).astype(int)
    # In case shape represents a vector, e.g. [[2, 2, 2], None]
    if shape[1] is None:
        shape[1] = np.ones(len(shape[0])).astype(int)

    shape = np.array(shape)

    def np2int(x):
        return list(map(int, x))

    tens = mat.view(*np2int(shape.flatten()))
    d = len(shape[0])
    # transpose_idx = 0, d, 1, d+1 ...
    transpose_idx = np.arange(2 * d).reshape(2, d).T.flatten()
    transpose_idx = np2int(transpose_idx)
    tens = tens.permute(*transpose_idx)
    new_shape = np2int(np.prod(shape, axis=0))
    tens = tens.contiguous().view(*new_shape)
    tt_tens = to_tt_tensor(tens, max_tt_rank, epsilon)
    tt_cores = []
    tt_ranks = tt_tens.ranks
    for core_idx in range(d):
        curr_core = tt_tens.tt_cores[core_idx]
        curr_rank = tt_ranks[core_idx]
        next_rank = tt_ranks[core_idx + 1]
        curr_core_new_shape = (curr_rank, shape[0, core_idx], shape[1, core_idx], next_rank)
        curr_core = curr_core.view(*curr_core_new_shape)
        tt_cores.append(curr_core)
    return TensorTrain(tt_cores, convert_to_tensors=False)


def orthogonalize_tt_cores(tt):
  """Orthogonalize TT-cores of a TT-object in the left to right order.
  Args:
    tt: TenosorTrain or a TensorTrainBatch.
  Returns:
    The same type as the input `tt` (TenosorTrain or a TensorTrainBatch).

  Complexity:
    for a single TT-object:
      O(d r^3 n)
    for a batch of TT-objects:
      O(batch_size d r^3 n)
    where
      d is the number of TT-cores (tt.ndims());
      r is the largest TT-rank of tt max(tt.get_tt_rank())
      n is the size of the axis dimension, e.g.
        for a tensor of size 4 x 4 x 4, n is 4;
        for a 9 x 64 matrix of raw shape (3, 3, 3) x (4, 4, 4) n is 12
  """
  # Left to right orthogonalization.
  ndims = tt.ndims
  raw_shape = tt.raw_shape
  tt_ranks = tt.ranks
  next_rank = tt_ranks[0]
  # Copy cores references so we can change the cores.
  tt_cores = list(tt.tt_cores)
  for core_idx in range(ndims - 1):
    curr_core = tt_cores[core_idx]
    # TT-ranks could have changed on the previous iteration, so `tt_ranks` can
    # be outdated for the current TT-rank, but should be valid for the next
    # TT-rank.
    curr_rank = next_rank
    next_rank = tt_ranks[core_idx + 1]
    if tt.is_tt_matrix:
      curr_mode_left = raw_shape[0][core_idx]
      curr_mode_right = raw_shape[1][core_idx]
      curr_mode = curr_mode_left * curr_mode_right
    else:
      curr_mode = raw_shape[core_idx]

    qr_shape = (curr_rank * curr_mode, next_rank)
    curr_core = torch.reshape(curr_core, qr_shape)
    curr_core, triang = torch.qr(curr_core)
    # if triang.get_shape().is_fully_defined():
    #   triang_shape = triang.get_shape().as_list()
    # else:
    #   triang_shape = torch.shape(triang)

    #Liancheng
    triang_shape = triang.shape


    # The TT-rank could have changed: if qr_shape is e.g. 4 x 10, than q would
    # be of size 4 x 4 and r would be 4 x 10, which means that the next rank
    # should be changed to 4.
    next_rank = triang_shape[0]
    if tt.is_tt_matrix:
      new_core_shape = (curr_rank, curr_mode_left, curr_mode_right, next_rank)
    else:
      new_core_shape = (curr_rank, curr_mode, next_rank)
    tt_cores[core_idx] = torch.reshape(curr_core, new_core_shape)

    next_core = torch.reshape(tt_cores[core_idx + 1], (triang_shape[1], -1))
    tt_cores[core_idx + 1] = torch.matmul(triang, next_core)

  if tt.is_tt_matrix:
    last_core_shape = (next_rank, raw_shape[0][-1], raw_shape[1][-1], 1)
  else:
    last_core_shape = (next_rank, raw_shape[-1], 1)
  tt_cores[-1] = torch.reshape(tt_cores[-1], last_core_shape)
  # TODO: infer the tt_ranks.
  return TensorTrain(tt_cores)




def round_tt(tt, max_tt_rank, epsilon=None):
    """Internal function that rounds a TensorTrain (not batch).
        See t3f.round for details.
    """
    ndims = tt.ndims
    max_tt_rank = np.array(max_tt_rank).astype(np.int32)
    if max_tt_rank < 1:
        raise ValueError('Maximum TT-rank should be greater or equal to 1.')
    if epsilon is not None and epsilon < 0:
        raise ValueError('Epsilon should be non-negative.')
    if max_tt_rank.size == 1:
        max_tt_rank = (max_tt_rank * np.ones(ndims + 1)).astype(np.int32)
    elif max_tt_rank.size != ndims + 1:
        raise ValueError('max_tt_rank should be a number or a vector of size (d+1) '
                     'where d is the number of dimensions (rank) of the tensor.')
    raw_shape = tt.raw_shape

    tt_cores = orthogonalize_tt_cores(tt).tt_cores
    # Copy cores references so we can change the cores.
    tt_cores = list(tt_cores)

    ranks = [1] * (ndims + 1)
    are_tt_ranks_defined = True
    #Right to left SVD compression.
    for core_idx in range(ndims - 1, 0, -1):
        curr_core = tt_cores[core_idx]
        if tt.is_tt_matrix:
            curr_mode_left = raw_shape[0][core_idx]
            curr_mode_right = raw_shape[1][core_idx]
            curr_mode = curr_mode_left * curr_mode_right
        else:
            curr_mode = raw_shape[core_idx]

        columns = curr_mode * ranks[core_idx + 1]
        curr_core = torch.reshape(curr_core, [-1, columns])
        #rows = curr_core.get_shape()[0].value
        #Liancheng
        rows = curr_core.shape[0]
        if rows is None:
            rows = torch.shape(curr_core)[0]
        if max_tt_rank[core_idx] == 1:
            ranks[core_idx] = 1
        else:
            try:
              ranks[core_idx] = min(max_tt_rank[core_idx], rows, columns)
            except TypeError:
                # Some of the values are undefined on the compilation stage and thus
                # they are tf.tensors instead of values.
                min_dim = torch.min(rows, columns)
                ranks[core_idx] = torch.min(max_tt_rank[core_idx], min_dim)
                are_tt_ranks_defined = False
        u, s, v = torch.svd(curr_core)
        u = u[:, 0:ranks[core_idx]]
        s = s[0:ranks[core_idx]]
        v = v[:, 0:ranks[core_idx]]
        if tt.is_tt_matrix:
            core_shape = (ranks[core_idx], curr_mode_left, curr_mode_right,
                    ranks[core_idx + 1])
        else:
            core_shape = (ranks[core_idx], curr_mode, ranks[core_idx + 1])
        tt_cores[core_idx] = torch.reshape(torch.t(v), core_shape)
        prev_core_shape = (-1, rows)
        tt_cores[core_idx - 1] = torch.reshape(tt_cores[core_idx - 1], prev_core_shape)
        tt_cores[core_idx - 1] = torch.matmul(tt_cores[core_idx - 1], u)
        tt_cores[core_idx - 1] = torch.matmul(tt_cores[core_idx - 1], torch.diag(s))

    if tt.is_tt_matrix:
        core_shape = (ranks[0], raw_shape[0][0], raw_shape[1][0], ranks[1])
    else:
        core_shape = (ranks[0], raw_shape[0], ranks[1])
    tt_cores[0] = torch.reshape(tt_cores[0], core_shape)
    if not are_tt_ranks_defined:
        ranks = None
    return TensorTrain(tt_cores)
