import numpy as np
import torch

from t3nsor.tensor_train import TensorTrain
from t3nsor.tensor_train import TensorTrainBatch
from t3nsor.utils import svd_fix
from numpy import linalg as LA
import pdb


def tt_to_ott(tt):
    tt_cores = tt.tt_cores
    ndims = tt.ndims
    shape = tt.raw_shape
    ott_cores = []
    #first core keep same
    ott_cores.append(tt_cores[0])
    #middle cores, do QR decomposition each slice
    for core_idx in range(1,ndims-1):
        for slice_idx in range(shape[core_idx]):
            tt_cores[core_idx][:,slice_idx,:] = torch.qr(tt_cores[core_idx][:,slice_idx,:])[0]
        ott_cores.append(tt_cores[core_idx])
    #last core, keep samp
    #pdb.set_trace()
    ott_cores.append(tt_cores[ndims-1])
    return TensorTrain(ott_cores, convert_to_tensors=False)


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


# TODO: rename round so not to shadow python.round?
def round(tt, max_tt_rank=None, epsilon=None, name='t3f_round'):
    """TT-rounding procedure, returns a TT object with smaller TT-ranks.
    Args:
      tt: `TensorTrain` object, TT-tensor or TT-matrix
      max_tt_rank: a number or a list of numbers
        If a number, than defines the maximal TT-rank of the result.
        If a list of numbers, than `max_tt_rank` length should be d+1
        (where d is the rank of `tens`) and `max_tt_rank[i]` defines
        the maximal (i+1)-th TT-rank of the result.
        The following two versions are equivalent
          `max_tt_rank = r`
        and
          `max_tt_rank = r * np.ones(d-1)`
      epsilon: a floating point number or None
        If the TT-ranks are not restricted (`max_tt_rank=np.inf`), then
        the result would be guarantied to be `epsilon` close to `tt`
        in terms of relative Frobenius error:
          ||res - tt||_F / ||tt||_F <= epsilon
        If the TT-ranks are restricted, providing a loose `epsilon` may
        reduce the TT-ranks of the result.
        E.g.
          round(tt, max_tt_rank=100, epsilon=0.9)
        will probably return you a TT-tensor with TT-ranks close to 1, not 100.
        Note that providing a nontrivial (= not equal to None) `epsilon` will make
        the TT-ranks of the result undefined on the compilation stage
        (e.g. res.get_tt_ranks() will return None, but t3f.tt_ranks(res).eval()
        will work).
      name: string, name of the Op.
    Returns:
      `TensorTrain` object containing a TT-tensor.
    Raises:
      ValueError if max_tt_rank is less than 0, if max_tt_rank is not a number and
        not a vector of length d + 1 where d is the number of dimensions (rank) of
        the input tensor, if epsilon is less than 0.
    """

    if isinstance(tt, TensorTrainBatch):
        return _round_batch_tt(tt, max_tt_rank, epsilon)
    else:
        return _round_tt(tt, max_tt_rank, epsilon)


def _round_tt(tt, max_tt_rank, epsilon):
    """Internal function that rounds a TensorTrain (not batch).
    See t3f.round for initial implementation.
    """
    ndims = tt.ndims
    max_tt_rank = np.array(max_tt_rank).astype(np.int32)
    if np.any(max_tt_rank < 1):
        raise ValueError('Maximum TT-rank should be greater or equal to 1.')
    if epsilon is not None and epsilon < 0:
        raise ValueError('Epsilon should be non-negative.')
    try:
        if len(max_tt_rank) != ndims + 1:
            raise ValueError('max_tt_rank should be a number or a vector of size (d+1) ''where d is the number of dimensions (rank) of the tensor.')
    except:
        #When max_tt_rank is a scalar
        max_tt_rank = (max_tt_rank * np.ones(ndims + 1)).astype(np.int32)

    raw_shape = tt.raw_shape

    tt, curr_core_list_orth = orthogonalize_tt_cores(tt)
    tt_cores = tt.tt_cores
    # # Copy cores references so we can change the cores.
    # tt_cores = list(tt_cores)

    ranks = [1] * (ndims + 1)

    curr_core_list = []
    # Right to left SVD compression.
    for core_idx in range(ndims - 1, 0, -1):

        curr_core = tt_cores[core_idx]
        if tt.is_tt_matrix:
            curr_mode_left = raw_shape[0][core_idx]
            curr_mode_right = raw_shape[1][core_idx]
            curr_mode = curr_mode_left * curr_mode_right
        else:
            #curr_mode = raw_shape[0][core_idx]
            curr_mode = raw_shape[core_idx]

        columns = curr_mode * ranks[core_idx + 1]
        curr_core = torch.reshape(curr_core, [-1, columns])
        rows = curr_core.shape[0]

        if max_tt_rank[core_idx] == 1:
            ranks[core_idx] = 1
        else:
            ranks[core_idx] = min(max_tt_rank[core_idx], rows, columns)
            # try:
            #     ranks[core_idx] = min(max_tt_rank[core_idx], rows, columns)
            # except TypeError:
            #     print('tpye error occurs')
            #     #in case max_tt_rank is not np.array, but torch.tensor
            #     min_dim = torch.min(rows, columns)
            #     ranks[core_idx] = torch.min(max_tt_rank[core_idx], min_dim)

        if LA.cond(curr_core.cpu().data.numpy()) > 1000:
            print("svd matrix cond num exploding at round_tt! Cond num:", LA.cond(curr_core.cpu().data.numpy()))
        curr_core_list.append(curr_core)

        uu, ss, vv = torch.svd(curr_core, some=True)

        u = uu[:, 0:ranks[core_idx]]
        s = ss[0:ranks[core_idx]]
        v = vv[:, 0:ranks[core_idx]]
        if tt.is_tt_matrix:
            core_shape = (ranks[core_idx], curr_mode_left, curr_mode_right,
                          ranks[core_idx + 1])
        else:
            core_shape = (ranks[core_idx], curr_mode, ranks[core_idx + 1])
        tt_cores[core_idx] = torch.reshape(torch.transpose(v, 0, 1), core_shape)
        prev_core_shape = (-1, rows)
        tt_cores[core_idx - 1] = torch.reshape(tt_cores[core_idx - 1], prev_core_shape)
        tt_cores[core_idx - 1] = torch.matmul(tt_cores[core_idx - 1], u)
        tt_cores[core_idx - 1] = torch.matmul(tt_cores[core_idx - 1], torch.diag(s))

    if tt.is_tt_matrix:
        core_shape = (ranks[0], raw_shape[0][0], raw_shape[1][0], ranks[1])
    else:
        core_shape = (ranks[0], raw_shape[0][0], ranks[1])
    tt_cores[0] = torch.reshape(tt_cores[0], core_shape)

    return TensorTrain(tt_cores), curr_core_list, curr_core_list_orth


def _round_batch_tt(tt, max_tt_rank, epsilon):
    """Internal function that rounds a TensorTrainBatch.
    See t3f.round for details.
    """
    ndims = tt.ndims
    max_tt_rank = np.array(max_tt_rank).astype(np.int32)
    if max_tt_rank < 1:
        raise ValueError('Maximum TT-rank should be greater or equal to 1.')
    if epsilon is not None and epsilon < 0:
        raise ValueError('Epsilon should be non-negative.')
    try:
        if len(max_tt_rank) != ndims + 1:
            raise ValueError('max_tt_rank should be a number or a vector of size (d+1) ''where d is the number of dimensions (rank) of the tensor.')
            max_tt_rank = (max_tt_rank * np.ones(ndims + 1)).astype(np.int32)
    except:
        max_tt_rank = (max_tt_rank * np.ones(ndims + 1)).astype(np.int32)
    raw_shape = tt.raw_shape
    batch_size = tt.batch_size

    tt_cores = orthogonalize_tt_cores(tt).tt_cores
    # Copy cores references so we can change the cores.
    tt_cores = list(tt_cores)

    ranks = [1] * (ndims + 1)
    are_tt_ranks_defined = True
    # Right to left SVD compression.
    for core_idx in range(ndims - 1, 0, -1):
        curr_core = tt_cores[core_idx]
        if tt.is_tt_matrix:
            curr_mode_left = raw_shape[0][core_idx]
            curr_mode_right = raw_shape[1][core_idx]
            curr_mode = curr_mode_left * curr_mode_right
        else:
            curr_mode = raw_shape[0][core_idx]

        columns = curr_mode * ranks[core_idx + 1]
        curr_core = torch.reshape(curr_core, (batch_size, -1, columns))
        rows = curr_core.shape[1]
        if rows is None:
            rows = curr_core.shape[1]
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

        #not calling
        #print("svd matrix cond num checking, line274: cond num:", LA.cond(curr_core.cpu().data.numpy()))

        s, u, v = torch.svd(curr_core, some=True)

        u = u[:, :, 0:ranks[core_idx]]
        s = s[:, 0:ranks[core_idx]]
        v = v[:, :, 0:ranks[core_idx]]
        if tt.is_tt_matrix:
            core_shape = (batch_size, ranks[core_idx], curr_mode_left, curr_mode_right,
                          ranks[core_idx + 1])
        else:
            core_shape = (batch_size, ranks[core_idx], curr_mode, ranks[core_idx + 1])
        tt_cores[core_idx] = torch.reshape(torch.transpose(v, (0, 2, 1)), core_shape)
        prev_core_shape = (batch_size, -1, rows)
        tt_cores[core_idx - 1] = torch.reshape(tt_cores[core_idx - 1], prev_core_shape)
        tt_cores[core_idx - 1] = torch.matmul(tt_cores[core_idx - 1], u)
        tt_cores[core_idx - 1] = torch.matmul(tt_cores[core_idx - 1], torch.diagflat(s))

    if tt.is_tt_matrix:
        core_shape = (batch_size, ranks[0], raw_shape[0][0], raw_shape[1][0], ranks[1])
    else:
        core_shape = (batch_size, ranks[0], raw_shape[0][0], ranks[1])
    tt_cores[0] = torch.reshape(tt_cores[0], core_shape)
    if not are_tt_ranks_defined:
        ranks = None
    return TensorTrainBatch(tt_cores, tt.raw_shape, ranks, batch_size=tt.batch_size)


def orthogonalize_tt_cores(tt, left_to_right=True,
                           name='t3f_orthogonalize_tt_cores'):
    """Orthogonalize TT-cores of a TT-object.
    Args:
      tt: TenosorTrain or a TensorTrainBatch.
      left_to_right: bool, the direction of orthogonalization.
      name: string, name of the Op.
    Returns:
      The same type as the input `tt` (TenosorTrain or a TensorTrainBatch).
    """

    if isinstance(tt, TensorTrainBatch):
        if left_to_right:
            return _orthogonalize_batch_tt_cores_left_to_right(tt)
        else:
            raise NotImplementedError('Batch right to left orthogonalization is ''not supported yet.')
    else:
        if left_to_right:
            return _orthogonalize_tt_cores_left_to_right(tt)
        else:
            return _orthogonalize_tt_cores_right_to_left(tt)


def _orthogonalize_tt_cores_left_to_right(tt):
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

    # store curr_core
    curr_core_list = []
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
            curr_mode = raw_shape[0][core_idx]

        qr_shape = (curr_rank * curr_mode, next_rank)
        curr_core = torch.reshape(curr_core, qr_shape)
        # curr_core, triang = torch.qr(curr_core, some=True)

        if LA.cond(curr_core.cpu().data.numpy()) > 1000:
            print("svd matrix cond num exploding at orthognalization! Cond num:", LA.cond(curr_core.cpu().data.numpy()))
        curr_core_list.append(curr_core)

        """pytorch does not support backprop of qr decomposition for flat matrix, now we naively use svd to approximate qr"""
        """a = qr,  a = usv^T, q = u, r = sv^T """
        uu, ss, vv = torch.svd(curr_core, some=True)
        curr_core = uu
        triang = torch.mm(torch.diag(ss), vv.permute(1,0))


        #if triang.shape[0] != triang.shape[1]:
        #    print("R is not a square matrix!")
        #if triang.get_shape().is_fully_defined():
        triang_shape = list(triang.shape)
        #else:
        #    triang_shape = tf.shape(triang)
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
        last_core_shape = (next_rank, raw_shape[0][-1], 1)
    tt_cores[-1] = torch.reshape(tt_cores[-1], last_core_shape)
    return TensorTrain(tt_cores, tt.raw_shape), curr_core_list


def _orthogonalize_batch_tt_cores_left_to_right(tt):
    """Orthogonalize TT-cores of a batch TT-object in the left to right order.
    Args:
      tt: TensorTrainBatch.
    Returns:
      TensorTrainBatch
    """
    # Left to right orthogonalization.
    ndims = tt.ndims
    raw_shape = tt.raw_shape
    tt_ranks = tt.tt_ranks
    next_rank = tt_ranks[0]
    batch_size = tt.batch_size

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
            curr_mode = raw_shape[0][core_idx]

        qr_shape = (batch_size, curr_rank * curr_mode, next_rank)
        curr_core = torch.reshape(curr_core, qr_shape)
        curr_core, triang = torch.qr(curr_core)
        # if triang.get_shape().is_fully_defined():
        #     triang_shape = triang.get_shape().as_list()
        # else:
        #     triang_shape = tf.shape(triang)
        triang_shape = list(triang.shape())
        # The TT-rank could have changed: if qr_shape is e.g. 4 x 10, than q would
        # be of size 4 x 4 and r would be 4 x 10, which means that the next rank
        # should be changed to 4.
        next_rank = triang_shape[1]
        if tt.is_tt_matrix:
            new_core_shape = (batch_size, curr_rank, curr_mode_left, curr_mode_right,
                              next_rank)
        else:
            new_core_shape = (batch_size, curr_rank, curr_mode, next_rank)

        tt_cores[core_idx] = torch.reshape(curr_core, new_core_shape)

        next_core = torch.reshape(tt_cores[core_idx + 1], (batch_size, triang_shape[2], -1))
        tt_cores[core_idx + 1] = torch.matmul(triang, next_core)

    if tt.is_tt_matrix:
        last_core_shape = (batch_size, next_rank, raw_shape[0][-1],
                           raw_shape[1][-1], 1)
    else:
        last_core_shape = (batch_size, next_rank, raw_shape[0][-1], 1)
    tt_cores[-1] = torch.reshape(tt_cores[-1], last_core_shape)

    return TensorTrainBatch(tt_cores, tt.raw_shape, batch_size=batch_size)


def _orthogonalize_tt_cores_right_to_left(tt):
    """Orthogonalize TT-cores of a TT-object in the right to left order.
    Args:
      tt: TenosorTrain or a TensorTrainBatch.
    Returns:
      The same type as the input `tt` (TenosorTrain or a TensorTrainBatch).
    """
    # Left to right orthogonalization.
    ndims = tt.ndims
    raw_shape = tt.raw_shape
    tt_ranks = tt.tt_ranks
    prev_rank = tt_ranks[ndims]
    # Copy cores references so we can change the cores.
    tt_cores = list(tt.tt_cores)
    for core_idx in range(ndims - 1, 0, -1):
        curr_core = tt_cores[core_idx]
        # TT-ranks could have changed on the previous iteration, so `tt_ranks` can
        # be outdated for the current TT-rank, but should be valid for the next
        # TT-rank.
        curr_rank = prev_rank
        prev_rank = tt_ranks[core_idx]
        if tt.is_tt_matrix:
            curr_mode_left = raw_shape[0][core_idx]
            curr_mode_right = raw_shape[1][core_idx]
            curr_mode = curr_mode_left * curr_mode_right
        else:
            curr_mode = raw_shape[0][core_idx]

        qr_shape = (prev_rank, curr_mode * curr_rank)
        curr_core = torch.reshape(curr_core, qr_shape)
        curr_core, triang = torch.qr(torch.transpose(curr_core, 0, 1))
        curr_core = torch.transpose(curr_core, 0, 1)
        triang = torch.transpose(triang)
        # if triang.get_shape().is_fully_defined():
        #     triang_shape = triang.get_shape().as_list()
        # else:
        #     triang_shape = tf.shape(triang)
        triang_shape = list(triang.shape)
        # The TT-rank could have changed: if qr_shape is e.g. 4 x 10, than q would
        # be of size 4 x 4 and r would be 4 x 10, which means that the next rank
        # should be changed to 4.
        prev_rank = triang_shape[1]
        if tt.is_tt_matrix:
            new_core_shape = (prev_rank, curr_mode_left, curr_mode_right, curr_rank)
        else:
            new_core_shape = (prev_rank, curr_mode, curr_rank)
        tt_cores[core_idx] = torch.reshape(curr_core, new_core_shape)

        prev_core = torch.reshape(tt_cores[core_idx - 1], (-1, triang_shape[0]))
        tt_cores[core_idx - 1] = torch.matmul(prev_core, triang)

    if tt.is_tt_matrix:
        first_core_shape = (1, raw_shape[0][0], raw_shape[1][0], prev_rank)
    else:
        first_core_shape = (1, raw_shape[0][0], prev_rank)
    tt_cores[0] = torch.reshape(tt_cores[0], first_core_shape)
    return TensorTrain(tt_cores, tt.raw_shape)
