import tensorflow as tf
import numpy as np
import t3f
from t3f.tensor_train_base import TensorTrainBase
from t3f.tensor_train import TensorTrain
from t3f.tensor_train_batch import TensorTrainBatch
from t3f import shapes
from t3f import utils
from t3f import decompositions
from t3f import initializers

def _validate_input_parameters(is_tensor, shape, **params):
  """Internal function for validating input parameters
  Args:
    is_tensor: bool, determines whether we attempt to construct a TT-tensor or
      a TT-matrix (needed for the correct shape checks).
    shape: array, the desired shape of the generated TT object
    params: optional, possible values:
      batch_size: int, for constructing batches
      tt_rank: array or int, desired TT-ranks
  """

  if is_tensor:
    if len(shape.shape) != 1:
      raise ValueError('shape should be 1d array, got %a' % shape)
    if np.any(shape < 1):
      raise ValueError('all elements in `shape` should be positive, got %a' %
                       shape)
    if not all(isinstance(sh, np.integer) for sh in shape):
      raise ValueError('all elements in `shape` should be integers, got %a' %
                       shape)
  else:
    if len(shape.shape) != 2:
      raise ValueError('shape should be 2d array, got %a' % shape)
    if shape[0].size != shape[1].size:
      raise ValueError('shape[0] should have the same length as shape[1], but'
                       'got %d and %d' % (shape[0].size, shape[1].size))
    if np.any(shape.flatten() < 1):
      raise ValueError('all elements in `shape` should be positive, got %a' %
                       shape)
    if not all(isinstance(sh, np.integer) for sh in shape.flatten()):
      raise ValueError('all elements in `shape` should be integers, got %a' %
                       shape)

  if 'batch_size' in params:
    batch_size = params['batch_size']
    if not isinstance(batch_size, (int, np.integer)):
      raise ValueError('`batch_size` should be integer, got %f' % batch_size)
    if batch_size < 1:
      raise ValueError('Batch size should be positive, got %d' % batch_size)
  if 'tt_rank' in params:
    tt_rank = params['tt_rank']
    if tt_rank.size == 1:
      if not isinstance(tt_rank[()], np.integer):
        raise ValueError('`tt_rank` should be integer, got %f' % tt_rank[()])
    if tt_rank.size > 1:
      if not all(isinstance(tt_r, np.integer) for tt_r in tt_rank):
        raise ValueError('all elements in `tt_rank` should be integers, got'
                         ' %a' % tt_rank)
    if np.any(tt_rank < 1):
      raise ValueError('`tt_rank` should be positive, got %a' % tt_rank)

    if is_tensor:
      if tt_rank.size != 1 and tt_rank.size != (shape.size + 1):
        raise ValueError('`tt_rank` array has inappropriate size, expected'
                         '1 or %d, got %d' % (shape.size + 1, tt_rank.size))
    else:
      if tt_rank.size != 1 and tt_rank.size != (shape[0].size + 1):
        raise ValueError('`tt_rank` array has inappropriate size, expected'
                         '1 or %d, got %d' % (shape[0].size + 1, tt_rank.size))


#initializer
def cp_like(shape, dtype=tf.float32, name='t3f_tensor_cp_like'):
  """Generate CP-like TT-tensor of the given shape, i.e TT rank = 1.
  Args:
    shape: array representing the shape of the future tensor
    dtype: [tf.float32] dtype of the resulting tensor.
    name: string, name of the Op.
  Returns:
    TensorTrain object containing a TT-tensor
  """

  shape = np.array(shape)
  _validate_input_parameters(is_tensor=True, shape=shape)
  num_dims = shape.size
  tt_rank = np.ones(num_dims + 1)

  with tf.name_scope(name):
    tt_cores = num_dims * [None]
    for i in range(num_dims):
      curr_core_shape = (1, shape[i], 1)
      tt_cores[i] = (1/shape[i])*tf.ones(curr_core_shape, dtype=dtype)

    return TensorTrain(tt_cores, shape, tt_rank)

#Compute Maximum norm of a tensor in TT-format in TT space
#eps: for t3f.round(tt,eps)
#max_rank: maximum rounding TT-rank
#TODO: shape can be infered from tt_tensor
def max_norm(tt_tensor,shape,eps=1e-6, max_iter=20,max_rank=140):
    y_0 = cp_like(shape)
    i=tf.constant(1.0)
    y_tuple_init = tuple([y_0.tt_cores[0],y_0.tt_cores[1],y_0.tt_cores[2],i])
    def cond(y_tt_cores1,y_tt_cores2,y_tt_cores3,i):
        return tf.less(i,max_iter)
    def body(y_tt_cores1,y_tt_cores2,y_tt_cores3,i):
        i = tf.add(i,1)
        y = TensorTrain([y_tt_cores1,y_tt_cores2,y_tt_cores3])
        q = t3f.multiply(tt_tensor,y)
        # print('type(q):',type(q))
        #z = tf.divide(q,tf.sqrt(t3f.flat_inner(q,q))
        z = t3f.multiply(q,tf.divide(tf.constant(1.0),tf.sqrt(t3f.flat_inner(q,q))))
        temp = t3f.round(z,max_rank,eps).tt_cores
        return tuple([temp[0],temp[1],temp[2],i])
    y_tuple_final = tf.while_loop(cond,
                                  body,
                                  y_tuple_init,
                                  shape_invariants=tuple([tf.TensorShape([1,shape[0],None]),tf.TensorShape([None,shape[1],None]),tf.TensorShape([None,shape[2],1]),i.get_shape()]))
    y_final = TensorTrain([y_tuple_final[0],y_tuple_final[1],y_tuple_final[2]])
    return tf.abs(t3f.flat_inner(y_final,t3f.multiply(tt_tensor, y_final)))


#Pointwise inverse
#Args: tt_tensor:
#      shape:
#      eps: rounding error
def point_inv(tt_tensor,shape, point_inv_aug, max_rank=140):
    # Choose u_tuple_init s.t condition holds
    eps=1e-3
    extra = tf.constant(1)
    #u_tuple_init = tuple(opt(tt_tensor, shape).tt_cores)
    #TODO: compute minima
    #use a ramdom start may cause problem
    #u_tuple_init = tuple(t3f.random_tensor((28,28,28),tt_rank=4).tt_cores)
    u_tuple_init = tuple(point_inv_aug.tt_cores)
    one_tt = t3f.tensor_ones(shape)
    #TODO:ugly
    def cond(u_tt_cores1,u_tt_cores2,u_tt_cores3):
        u_tt = TensorTrain([u_tt_cores1,u_tt_cores2,u_tt_cores3])
        return tf.less(eps*max_norm(tt_tensor,shape), max_norm(one_tt - t3f.multiply(tt_tensor,u_tt),shape))

    def body(u_tt_cores1,u_tt_cores2,u_tt_cores3):
        u_tt = TensorTrain([u_tt_cores1,u_tt_cores2,u_tt_cores3])
        return tuple(t3f.round(t3f.multiply(u_tt, 2*one_tt-t3f.multiply(tt_tensor, u_tt)), max_rank,eps).tt_cores)

    #c = lambda (u_tt,): tf.less(eps*t3f.frobenius_norm(tt_tensor), t3f.frobenius_norm(one_tt - t3f.multiply(tt_tensor,u_tt)))
    #b = lambda (u_tt,): (t3f.round(t3f.multiply(u_tt, 2*one_tt-t3f.multiply(tt_tensor, u_tt))),)
    u_tuple_final = tf.while_loop(cond,
                                  body,
                                  u_tuple_init,
                                  shape_invariants=tuple([tf.TensorShape([1,shape[0],None]),tf.TensorShape([None,shape[1],None]),tf.TensorShape([None,shape[2],1])]))
    return TensorTrain(u_tuple_final)


#sign
def sign(tt_tensor, shape, point_inv_aug,eps,max_rank=40):
    eps = 1e-3
    # print("eps_type111111111:",type(eps))
    # print("max_rank:",type(max_rank))
    one_tt = t3f.tensor_ones(shape)
    #extra = tf.constant(1)
    #TODO: tt_tensor is a tuple of tf.tensor, how to pass a tuple of tf.tensor to tf.while_loop
    u_tuple_init = tuple(tt_tensor.tt_cores)
    # print("u_tuple_init1 :", u_tuple_init)
    # print("type:",type(u_tuple_init))
    # print("core1:",u_tuple_init[0])

    #Args:u_tt_cores is tuple of tf.tensor
    #let num = len(tt_tensor.tt_cores)
    #TODO:ugly
    def cond(u_tt_cores1, u_tt_cores2, u_tt_cores3):
        #TODO:use u_tt_cores to reconstuct tt_tensor
        #TensorTrain class should be able to infer the shape from tt_cores
        u_tt = TensorTrain([u_tt_cores1, u_tt_cores2, u_tt_cores3])
        # print("cond",u_tt)
        # print("eps_type:",type(eps))
        # print("22222",type(max_norm(tt_tensor,shape)))
        # print("1111",type(eps*max_norm(tt_tensor,shape)))
        return tf.less(eps*max_norm(tt_tensor,shape),
                       max_norm(one_tt-t3f.multiply(u_tt,u_tt),shape))

    def body(u_tt_cores1, u_tt_cores2, u_tt_cores3):
        u_tt = TensorTrain([u_tt_cores1, u_tt_cores2, u_tt_cores3])
        # print("body",u_tt)
        return tuple(tf.cond(tf.less(max_norm(one_tt - t3f.multiply(u_tt,u_tt),shape),tf.constant(1,dtype=tf.float32)),
                      lambda: t3f.round(1/2 * t3f.multiply(u_tt, 3*one_tt - t3f.multiply(u_tt, u_tt)),max_rank,eps).tt_cores,
                      lambda: t3f.round(1/2 * (u_tt + point_inv(u_tt, shape, point_inv_aug)),max_rank,eps).tt_cores ))

    u_tuple_final = tf.while_loop(cond,
                                  body,
                                  u_tuple_init,
                                  shape_invariants=tuple([tf.TensorShape([1,shape[0],None]),tf.TensorShape([None,shape[1],None]),tf.TensorShape([None,shape[2],1])]))
    print('print inv:', TensorTrain(u_tuple_final))
    return TensorTrain(u_tuple_final)

#characteristic
def characteristic(tt_tensor, shape, eps, point_inv_aug):
    one_tt = t3f.tensor_ones(shape)
    max_rank = 140
    err = 1e-6
    print('print sign',sign(-1 * tt_tensor, shape, point_inv_aug, eps, max_rank=140))
    one_cores = tuple(one_tt.tt_cores)
    one_tt_tensortrain = TensorTrain(one_cores)
    #return t3f.round(1/2 * (one_tt_tensortrain - sign(-1 * tt_tensor, shape, point_inv_aug, eps, max_rank=140)),max_rank, eps)
    return 1/2*(one_tt_tensortrain + sign(tt_tensor, shape, point_inv_aug, eps))

#level set
def level_set(tt_tensor, shape, eps, point_inv_aug):
    return t3f.multiply(characteristic(tt_tensor, shape, eps, point_inv_aug),tt_tensor)
