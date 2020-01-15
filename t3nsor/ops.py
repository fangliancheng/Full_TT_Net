from t3nsor import TensorTrainBatch
from t3nsor import TensorTrain
import t3nsor as t3
import torch
from t3nsor.utils import kronecker_product
import pdb

def cayley(t):
    #
    for i in t.shape[0]:
        t[i,:,:] = torch.matmul( torch.eye(t.shape[1],t.shape[2]) - t[i,:,:] , torch.inverse(torch.eye(t.shape[1],t.shape[2]) + t[i,:,:]) )
    return t

#TODO: multiplication with a scalar
def scalar_tt_mul(tt,r):
    first_core = r*tt.tt_cores[0]
    new_cores = []
    new_cores.append(first_core)
    for i in range(1,tt.ndims):
        new_cores.append(tt.tt_cores[i])
    return TensorTrain(new_cores)


#dot product, will be called to compute frobenius_norm
def tt_tt_flat_inner(tt_a, tt_b):
  """Inner product between two TT-tensors or TT-matrices along all axis.
  The shapes of tt_a and tt_b should coincide.
  Args:
    tt_a: `TensorTrain` or `TensorTrainBatch` object
    tt_b: `TensorTrain` or `TensorTrainBatch` object
  Returns
    a number or a Tensor with numbers for each element in the batch.
    sum of products of all the elements of tt_a and tt_b
  Raises:
    ValueError if the arguments are not `TensorTrain` objects, have different
      number of TT-cores, different underlying shape, or if you are trying to
      compute inner product between a TT-matrix and a TT-tensor.
  Complexity:
    Multiplying two single TT-objects is O(d r^3 n) where d is the number of
      TT-cores (tt_a.ndims()), r is the largest TT-rank
        max(tt_a.get_tt_rank(), tt_b.get_tt_rank())
      and n is the size of the axis dimension, e.g.
        for a tensor of size 4 x 4 x 4, n is 4;
        for a 9 x 64 matrix of raw shape (3, 3, 3) x (4, 4, 4) n is 12
      A more precise complexity is O(d r1 r2 n max(r1, r2)) where
        r1 is the largest TT-rank of tt_a and r2 is the largest TT-rank of tt_b.
    The complexity of this operation for batch input is O(batch_size d r^3 n).
  """

  if tt_a.is_tt_matrix() != tt_b.is_tt_matrix():
    raise ValueError('One of the arguments is a TT-tensor, the other is '
                     'a TT-matrix, disallowed')
  are_both_matrices = tt_a.is_tt_matrix() and tt_b.is_tt_matrix()

  # TODO: compare shapes and raise if not consistent.

  ndims = tt_a.ndims
  if tt_b.ndims != ndims:
    raise ValueError('Arguments should have the same number of dimensions, '
                     'got %d and %d instead.' % (ndims, tt_b.ndims()))

  axes_str = 'ij' if are_both_matrices else 'i'
  # Convert BatchSize 1 batch into TT object to simplify broadcasting.
  # tt_a = shapes.squeeze_batch_dim(tt_a)
  # tt_b = shapes.squeeze_batch_dim(tt_b)
  is_a_batch = isinstance(tt_a, TensorTrainBatch)
  is_b_batch = isinstance(tt_b, TensorTrainBatch)
  is_res_batch = is_a_batch or is_b_batch
  a_batch_str = 'o' if is_a_batch else ''
  b_batch_str = 'o' if is_b_batch else ''
  res_batch_str = 'o' if is_res_batch else ''
  init_einsum_str = '{1}a{0}b,{2}c{0}d->{3}bd'.format(axes_str, a_batch_str,
                                                      b_batch_str,
                                                      res_batch_str)
  a_core = tt_a.tt_cores[0]
  b_core = tt_b.tt_cores[0]
  # Simplest example of this operation:
  # if both arguments are TT-tensors, then it is
  # res = tf.einsum('aib,cid->bd', a_core, b_core)
  res = torch.einsum(init_einsum_str, a_core, b_core)

  einsum_str = '{3}ac,{1}a{0}b,{2}c{0}d->{3}bd'.format(axes_str, a_batch_str,
                                                       b_batch_str,
                                                       res_batch_str)
  for core_idx in range(1, ndims):
    a_core = tt_a.tt_cores[core_idx]
    b_core = tt_b.tt_cores[core_idx]
    # Simplest example of this operation:
    # if both arguments are TT-tensors, then it is
    # res = tf.einsum('ac,aib,cid->bd', res, a_core, b_core)
    res = torch.einsum(einsum_str, res, a_core, b_core)
  return torch.squeeze(res)


def frobenius_norm_squared(tt, differentiable=True):
    """
    Args:'TensorTrain' or 'TensorTrainBatch' objsect
    TODO:

    Returns
    a number which is the Frobenius norm squared of `tt`, if it is `TensorTrain`
    OR
    a Tensor of size tt.batch_size, consisting of the Frobenius norms squared of
    each TensorTrain in `tt`, if it is `TensorTrainBatch`
    """

    if differentiable:
        if isinstance(tt, TensorTrainBatch):
            bs_str = 'n'
        else:
            bs_str = ''
        if tt.is_tt_matrix:
            running_prod = torch.einsum('{0}aijb,{0}cijd->{0}bd'.format(bs_str),
                                 tt.tt_cores[0], tt.tt_cores[0])
        else:
            running_prod = torch.einsum('{0}aib,{0}cid->{0}bd'.format(bs_str),
                                 tt.tt_cores[0], tt.tt_cores[0])

        for core_idx in range(1, tt.ndims):
            curr_core = tt.tt_cores[core_idx]
            if tt.is_tt_matrix:
                running_prod = torch.einsum('{0}ac,{0}aijb,{0}cijd->{0}bd'.format(bs_str),
                                   running_prod, curr_core, curr_core)
            else:
                running_prod = torch.einsum('{0}ac,{0}aib,{0}cid->{0}bd'.format(bs_str),
                                   running_prod, curr_core, curr_core)

        return torch.squeeze(torch.squeeze(running_prod, dim=-1), dim=-1)

    else:
      # orth_tt = decompositions.orthogonalize_tt_cores(tt, left_to_right=True)
      # # All the cores of orth_tt except the last one are orthogonal, hence
      # # the Frobenius norm of orth_tt equals to the norm of the last core.
      # if hasattr(tt, 'batch_size'):
      #   batch_size = shapes.lazy_batch_size(tt)
      #   last_core = tf.reshape(orth_tt.tt_cores[-1], (batch_size, -1))
      #   return tf.norm(last_core, axis=1) ** 2
      # else:
      #   return tf.norm(orth_tt.tt_cores[-1]) ** 2
        raise NotImplementedError


#hadmard product
def multiply(tt_left, tt_right):
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


def _add_tensor_cores(tt_a, tt_b):
  """Internal function to be called from add for two TT-tensors.
  Does the actual assembling of the TT-cores to add two TT-tensors.
  """
    ndims = tt_a.ndims
    dtype = tt_a.dtype
    shape = tt_a.raw_shape
    a_ranks = tt_a.tt_ranks
    b_ranks = tt_b.tt_ranks
    tt_cores = []
    for core_idx in range(ndims):
        a_core = tt_a.tt_cores[core_idx]
        b_core = tt_b.tt_cores[core_idx]
        if core_idx == 0:
            curr_core = torch.cat((a_core, b_core), dim=2)
        elif core_idx == ndims - 1:
            curr_core = torch.cat((a_core, b_core), dim=0)
        else:
            upper_zeros = torch.zeros((a_ranks[core_idx], shape[0][core_idx], b_ranks[core_idx + 1]), dtype)
            lower_zeros = torch.zeros((b_ranks[core_idx], shape[0][core_idx], a_ranks[core_idx + 1]), dtype)
            upper = torch.cat((a_core, upper_zeros), dim=2)
            lower = torch.cat((lower_zeros, b_core), dim=2)
            curr_core = torch.cat((upper, lower), dim=0)
        tt_cores.append(curr_core)
    return tt_cores


def _add_batch_tensor_cores(tt_a, tt_b):
  """Internal function to be called from add for two batches of TT-tensors.
  Does the actual assembling of the TT-cores to add two batches of TT-tensors.
  """
    ndims = tt_a.ndims
    dtype = tt_a.dtype
    shape = tt_a.raw_shape
    a_ranks = tt_a.tt_ranks
    b_ranks = tt_b.tt_ranks
    if isinstance(tt_a, TensorTrainBatch) and tt_a.batch_size == 1:
    # We add 1 element batch tt_a to a batch_size element batch tt_b to get
    # the answer TensorTrainBatch of batch_size == tt_b.batch_size.
        batch_size = tt_b.batch_size
    else:
        batch_size = tt_a.batch_size
    tt_a = t3.utils.expand_batch_dim(tt_a)
    tt_b = t3.utils.expand_batch_dim(tt_b)
    tt_cores = []
    for core_idx in range(ndims):
        a_core = tt_a.tt_cores[core_idx]
        if tt_a.batch_size == 1:
            a_core = a_core.repeat(batch_size, 1, 1, 1)
        b_core = tt_b.tt_cores[core_idx]
        if tt_b.batch_size == 1:
            b_core = b_core.repeat(batch_size, 1, 1, 1)
        if core_idx == 0:
            curr_core = torch.cat((a_core, b_core), dim=3)
        elif core_idx == ndims - 1:
            curr_core = torch.cat((a_core, b_core), dim=1)
        else:
            upper_zeros = torch.zeros((batch_size, a_ranks[core_idx], shape[0][core_idx],
                              b_ranks[core_idx + 1]), dtype)
            lower_zeros = torch.zeros((batch_size, b_ranks[core_idx], shape[0][core_idx],
                              a_ranks[core_idx + 1]), dtype)
            upper = torch.cat((a_core, upper_zeros), dim=3)
            lower = torch.cat((lower_zeros, b_core), dim=3)
            curr_core = torch.cat((upper, lower), dim=1)
        tt_cores.append(curr_core)
    return tt_cores, batch_size


def _add_matrix_cores(tt_a, tt_b):
  """Internal function to be called from add for two TT-matrices.
  Does the actual assembling of the TT-cores to add two TT-matrices.
  """
  ndims = tt_a.ndims
  dtype = tt_a.dtype
  shape = tt_a.raw_shape
  a_ranks = tt_a.tt_ranks
  b_ranks = tt_b.tt_ranks
  tt_cores = []
  for core_idx in range(ndims):
    a_core = tt_a.tt_cores[core_idx]
    b_core = tt_b.tt_cores[core_idx]
    if core_idx == 0:
      curr_core = torch.cat((a_core, b_core), dim=3)
    elif core_idx == ndims - 1:
      curr_core = torch.cat((a_core, b_core), dim=0)
    else:
      upper_zeros = torch.zeros((a_ranks[core_idx], shape[0][core_idx],
                              shape[1][core_idx], b_ranks[core_idx + 1]), dtype)
      lower_zeros = torch.zeros((b_ranks[core_idx], shape[0][core_idx],
                              shape[1][core_idx], a_ranks[core_idx + 1]), dtype)
      upper = torch.cat((a_core, upper_zeros), dim=3)
      lower = torch.cat((lower_zeros, b_core), dim=3)
      curr_core = torch.cat((upper, lower), dim=0)
    tt_cores.append(curr_core)
  return tt_cores


def _add_batch_matrix_cores(tt_a, tt_b):
  """Internal function to be called from add for two batches of TT-matrices.
  Does the actual assembling of the TT-cores to add two batches of TT-matrices.
  """
  ndims = tt_a.ndims
  dtype = tt_a.dtype
  shape = tt_a.raw_shape
  a_ranks = tt_a.tt_ranks
  b_ranks = tt_b.tt_ranks
  if isinstance(tt_a, TensorTrainBatch) and tt_a.batch_size == 1:
    # We add 1 element batch tt_a to a batch_size element batch tt_b to get
    # the answer TensorTrainBatch of batch_size == tt_b.batch_size.
    batch_size = tt_b.batch_size
  else:
    batch_size = tt_a.batch_size
  tt_a = t3.utils.expand_batch_dim(tt_a)
  tt_b = t3.utils.expand_batch_dim(tt_b)
  tt_cores = []
  for core_idx in range(ndims):
    a_core = tt_a.tt_cores[core_idx]
    if tt_a.batch_size == 1:
      a_core = a_core.repeat(batch_size, 1, 1, 1, 1)
    b_core = tt_b.tt_cores[core_idx]
    if tt_b.batch_size == 1:
      b_core = b_core.repeat(batch_size, 1, 1, 1, 1)
    if core_idx == 0:
      curr_core = torch.cat((a_core, b_core), dim=4)
    elif core_idx == ndims - 1:
      curr_core = torch.cat((a_core, b_core), dim=1)
    else:
      upper_zeros = torch.zeros((batch_size, a_ranks[core_idx], shape[0][core_idx],
                              shape[1][core_idx], b_ranks[core_idx + 1]), dtype)
      lower_zeros = torch.zeros((batch_size, b_ranks[core_idx], shape[0][core_idx],
                              shape[1][core_idx], a_ranks[core_idx + 1]), dtype)
      upper = torch.cat((a_core, upper_zeros), dim=4)
      lower = torch.cat((lower_zeros, b_core), dim=4)
      curr_core = torch.cat((upper, lower), dim=1)
    tt_cores.append(curr_core)
  return tt_cores, batch_size


def add(tt_a, tt_b, name='t3f_add'):
  """Returns a TensorTrain corresponding to elementwise sum tt_a + tt_b.
  The shapes of tt_a and tt_b should coincide.
  Supports broadcasting:
    add(TensorTrainBatch, TensorTrain)
  adds TensorTrain to each element in the batch of TTs in TensorTrainBatch.
  Args:
    tt_a: `TensorTrain`, `TensorTrainBatch`, TT-tensor, or TT-matrix
    tt_b: `TensorTrain`, `TensorTrainBatch`, TT-tensor, or TT-matrix
    name: string, name of the Op.
  Returns
    a `TensorTrain` object corresponding to the element-wise sum of arguments if
      both arguments are `TensorTrain`s.
    OR a `TensorTrainBatch` if at least one of the arguments is
      `TensorTrainBatch`
  Raises
    ValueError if the arguments shapes do not coincide
  """
    ndims = tt_a.ndims
    if tt_a.is_tt_matrix() != tt_b.is_tt_matrix():
        raise ValueError('The arguments should be both TT-tensors or both '
                     'TT-matrices')

    if tt_a.raw_shape != tt_b.raw_shape:
        raise ValueError('The arguments should have the same shape.')

    if not t3.utils.is_batch_broadcasting_possible(tt_a, tt_b):
        raise ValueError('The batch sizes are different and not 1, broadcasting is '
                     'not available.')

    is_a_batch = isinstance(tt_a, TensorTrainBatch)
    is_b_batch = isinstance(tt_b, TensorTrainBatch)
    is_batch_case = is_a_batch or is_b_batch
    batch_size = None
    if is_batch_case:
      if tt_a.is_tt_matrix():
        tt_cores, batch_size = _add_batch_matrix_cores(tt_a, tt_b)
      else:
        tt_cores, batch_size = _add_batch_tensor_cores(tt_a, tt_b)
    else:
      if tt_a.is_tt_matrix():
        tt_cores = _add_matrix_cores(tt_a, tt_b)
      else:
        tt_cores = _add_tensor_cores(tt_a, tt_b)

    out_ranks = [1]
    static_a_ranks = tt_a.tt_ranks
    static_b_ranks = tt_b.tt_ranks
    for core_idx in range(1, ndims):
      out_ranks.append(static_a_ranks[core_idx] + static_b_ranks[core_idx])
    out_ranks.append(1)
    if is_batch_case:
      return TensorTrainBatch(tt_cores, tt_a.raw_shape, out_ranks,batch_size)
    else:
      return TensorTrain(tt_cores, tt_a.raw_shape, out_ranks)



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


def transpose(tt_matrix,convert_to_tensors=False):
    cores = []
    for core in tt_matrix.tt_cores:
        #print('core device',core.device)
        cores.append(core.transpose(1, 2))
    return TensorTrain(cores,convert_to_tensors=convert_to_tensors)


#Liancheng
def tt_tt_matmul(tt_matrix_a, tt_matrix_b, convert_to_tensors=False):
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
    ValueError if the arguments are not TT matrices or if their sizes are not
    appropriate for a matrix-by-matrix multiplication.
  """

  #   if not isinstance(tt_matrix_a, TensorTrainBase) or \
  #       not isinstance(tt_matrix_b, TensorTrainBase) or \
  #       not tt_matrix_a.is_tt_matrix() or \
  #       not tt_matrix_b.is_tt_matrix():
  #       raise ValueError('Arguments should be TT-matrices')
  #
    if not t3.utils.is_batch_broadcasting_possible(tt_matrix_a, tt_matrix_b):
        raise ValueError('The batch sizes are different and not 1, broadcasting is '
                     'not available.')

    ndims = tt_matrix_a.ndims
    if tt_matrix_b.ndims != ndims:
        raise ValueError('Arguments should have the same number of dimensions, '
                     'got %d and %d instead.' % (ndims, tt_matrix_b.ndims))

    #Convert BatchSize 1 batch into TT object to simplify broadcasting.
    tt_matrix_a = t3.utils.squeeze_batch_dim(tt_matrix_a)
    tt_matrix_b = t3.utils.squeeze_batch_dim(tt_matrix_b)
    is_a_batch = isinstance(tt_matrix_a, TensorTrainBatch)
    is_b_batch = isinstance(tt_matrix_b, TensorTrainBatch)
    is_res_batch = is_a_batch or is_b_batch
    a_batch_str = 'o' if is_a_batch else ''
    b_batch_str = 'o' if is_b_batch else ''
    res_batch_str = 'o' if is_res_batch else ''
    einsum_str = '{}aijb,{}cjkd->{}acikbd'.format(a_batch_str, b_batch_str,
                                                 res_batch_str)
    result_cores = []
    a_shape = tt_matrix_a.raw_shape
    a_ranks = tt_matrix_a.ranks
    b_shape = tt_matrix_b.raw_shape
    b_ranks = tt_matrix_b.ranks

    if is_res_batch:
        if is_a_batch:
            batch_size = tt_matrix_a.tt_cores[0].shape[0]
        if is_b_batch:
            batch_size = tt_matrix_b.tt_cores[0].shape[0]

    for core_idx in range(ndims):
        a_core = tt_matrix_a.tt_cores[core_idx]
        b_core = tt_matrix_b.tt_cores[core_idx]

        curr_res_core = torch.einsum(einsum_str, a_core, b_core)

        res_left_rank = a_ranks[core_idx] * b_ranks[core_idx]
        res_right_rank = a_ranks[core_idx + 1] * b_ranks[core_idx + 1]
        left_mode = a_shape[0][core_idx]
        right_mode = b_shape[1][core_idx]
        if is_res_batch:
            core_shape = (batch_size, res_left_rank, left_mode, right_mode, res_right_rank)
        else:
            core_shape = (res_left_rank, left_mode, right_mode, res_right_rank)
        curr_res_core = torch.reshape(curr_res_core, core_shape)
        result_cores.append(curr_res_core)

    res_shape = (tt_matrix_a.raw_shape[0], tt_matrix_b.raw_shape[1])
    static_a_ranks = tt_matrix_a.ranks
    static_b_ranks = tt_matrix_b.ranks
    out_ranks = [a_r * b_r for a_r, b_r in zip(static_a_ranks, static_b_ranks)]
    if is_res_batch:
        return TensorTrainBatch(result_cores, convert_to_tensors=convert_to_tensors)
    else:
        return TensorTrain(result_cores, convert_to_tensors=convert_to_tensors)


#TODO: add dense_tt_matmul
def tt_dense_matmul(tt_matrix_a, matrix_b,convert_to_tensors=False):
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
            raise ValueError('Arguments shapes should align got %d and %d instead.' %(tt_matrix_a.shape, matrix_b.shape))

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
        #print('curr_core device:',curr_core.device)

        #Liancheng
        curr_core = curr_core.to(data.device)

        # On the k = core_idx iteration, after applying einsum the shape of data
        # becomes ik x (ik-1..., id-1, K, j0, ..., jk-1) x rank_k
        #print('curr_core device:',curr_core.device)
        #print('data device:',data.device)
        data = torch.einsum('aijb,rjb->ira', curr_core, data)
        if core_idx > 0:
          # After reshape the shape of data becomes
          # (ik, ..., id-1, K, j0, ..., jk-2) x jk-1 x rank_k
            new_data_shape = (-1, a_raw_shape[1][core_idx - 1], a_ranks[core_idx])
            data = data.contiguous().view(new_data_shape)

    # At the end the shape of the data is (i0, ..., id-1) x K
    return data.view(a_shape[0], b_shape[1])
