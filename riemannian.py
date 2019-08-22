from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import t3f
import matplotlib.pyplot as plt
import pylab
from t3f.tensor_train import TensorTrain
from t3f.tensor_train_batch import TensorTrainBatch
from t3f.tensor_train_base import TensorTrainBase
from t3f import utils
import time
from t3f import shapes
from test_misc import self_multiply

print("GPU Available: ", tf.test.is_gpu_available())
print(tf.test.gpu_device_name())


def multiply(tt_left, right, name='t3f_multiply'):
  """Returns a TensorTrain corresponding to element-wise product tt_left * right.
  Supports broadcasting:
    multiply(TensorTrainBatch, TensorTrain) returns TensorTrainBatch consisting
    of element-wise products of TT in TensorTrainBatch and TensorTrain
    multiply(TensorTrainBatch_a, TensorTrainBatch_b) returns TensorTrainBatch
    consisting of element-wise products of TT in TensorTrainBatch_a and
    TT in TensorTrainBatch_b
    Batch sizes should support broadcasting
  Args:
    tt_left: `TensorTrain` OR `TensorTrainBatch`
    right: `TensorTrain` OR `TensorTrainBatch` OR a number.
    name: string, name of the Op.
  Returns
    a `TensorTrain` or `TensorTrainBatch` object corresponding to the
    element-wise product of the arguments.
  Raises
    ValueError if the arguments shapes do not coincide or broadcasting is not
    possible.
  """
  print('multipy function calling')
  is_left_batch = isinstance(tt_left, TensorTrainBatch)
  is_right_batch = isinstance(right, TensorTrainBatch)

  is_batch_case = is_left_batch or is_right_batch
  ndims = tt_left.ndims()
  print('ndims',ndims)
  if not isinstance(right, TensorTrainBase):
    with tf.name_scope(name, values=tt_left.tt_cores+(right,)):
      # Assume right is a number, not TensorTrain.
      # To squash right uniformly across TT-cores we pull its absolute value
      # and raise to the power 1/ndims. First TT-core is multiplied by the sign
      # of right.
      tt_cores = list(tt_left.tt_cores)
      fact = tf.pow(tf.cast(tf.abs(right), tt_left.dtype), 1.0 / ndims)
      sign = tf.cast(tf.sign(right), tt_left.dtype)
      for i in range(len(tt_cores)):
        tt_cores[i] = fact * tt_cores[i]

      tt_cores[0] = tt_cores[0] * sign
      out_ranks = tt_left.get_tt_ranks()
      if is_left_batch:
          out_batch_size = tt_left.batch_size
  else:
    with tf.name_scope(name, values=tt_left.tt_cores+right.tt_cores):

      if tt_left.is_tt_matrix() != right.is_tt_matrix():
        raise ValueError('The arguments should be both TT-tensors or both '
                         'TT-matrices')

      if tt_left.get_raw_shape() != right.get_raw_shape():
        raise ValueError('The arguments should have the same shape.')

      out_batch_size = 1
      dependencies = []
      can_determine_if_broadcast = True
      if is_left_batch and is_right_batch:
        if tt_left.batch_size is None and right.batch_size is None:
          can_determine_if_broadcast = False
        elif tt_left.batch_size is None and right.batch_size is not None:
          if right.batch_size > 1:
              can_determine_if_broadcast = False
        elif tt_left.batch_size is not None and right.batch_size is None:
          if tt_left.batch_size > 1:
              can_determine_if_broadcast = False

      if not can_determine_if_broadcast:
        # Cannot determine if broadcasting is needed. Avoid broadcasting and
        # assume elementwise multiplication AND add execution time assert to
        # print a better error message if the batch sizes turn out to be
        # different.

        message = ('The batch sizes were unknown on compilation stage, so '
                   'assumed elementwise multiplication (i.e. no broadcasting). '
                   'Now it seems that they are different after all :')

        data = [message, shapes.lazy_batch_size(tt_left), ' x ',
                shapes.lazy_batch_size(right)]
        bs_eq = tf.assert_equal(shapes.lazy_batch_size(tt_left),
                                shapes.lazy_batch_size(right), data=data)

        dependencies.append(bs_eq)

      do_broadcast = shapes.is_batch_broadcasting_possible(tt_left, right)
      if not can_determine_if_broadcast:
        # Assume elementwise multiplication if broadcasting cannot be determined
        # on compilation stage.
        do_broadcast = False
      if not do_broadcast and can_determine_if_broadcast:
        raise ValueError('The batch sizes are different and not 1, broadcasting '
                         'is not available.')

      a_ranks = shapes.lazy_tt_ranks(tt_left)
      b_ranks = shapes.lazy_tt_ranks(right)
      shape = shapes.lazy_raw_shape(tt_left)

      output_str = ''
      bs_str_left = ''
      bs_str_right = ''

      if is_batch_case:
        if is_left_batch and is_right_batch:
          # Both arguments are batches of equal size.
          if tt_left.batch_size == right.batch_size or not can_determine_if_broadcast:
            bs_str_left = 'n'
            bs_str_right = 'n'
            output_str = 'n'
            if not can_determine_if_broadcast:
              out_batch_size = None
            else:
              out_batch_size = tt_left.batch_size
          else:
            # Broadcasting (e.g batch_sizes are 1 and n>1).
            bs_str_left = 'n'
            bs_str_right = 'm'
            output_str = 'nm'
            if tt_left.batch_size is None or tt_left.batch_size > 1:
              out_batch_size = tt_left.batch_size
            else:
              out_batch_size = right.batch_size
        else:
          # One of the arguments is TensorTrain.
          if is_left_batch:
            bs_str_left = 'n'
            bs_str_right = ''
            out_batch_size = tt_left.batch_size
          else:
            bs_str_left = ''
            bs_str_right = 'n'
            out_batch_size = right.batch_size
          output_str = 'n'

      is_matrix = tt_left.is_tt_matrix()
      tt_cores = []

      for core_idx in range(ndims):
        a_core = tt_left.tt_cores[core_idx]
        b_core = right.tt_cores[core_idx]
        left_rank = a_ranks[core_idx] * b_ranks[core_idx]
        right_rank = a_ranks[core_idx + 1] * b_ranks[core_idx + 1]
        if is_matrix:
          with tf.control_dependencies(dependencies):
            curr_core = tf.einsum('{0}aijb,{1}cijd->{2}acijbd'.format(bs_str_left,
                                  bs_str_right, output_str), a_core, b_core)
            curr_core = tf.reshape(curr_core, (-1, left_rank,
                                               shape[0][core_idx],
                                               shape[1][core_idx],
                                               right_rank))
            if not is_batch_case:
                curr_core = tf.squeeze(curr_core, axis=0)
        else:
          with tf.control_dependencies(dependencies):
            curr_core = tf.einsum('{0}aib,{1}cid->{2}acibd'.format(bs_str_left,
                                  bs_str_right, output_str), a_core, b_core)
            curr_core = tf.reshape(curr_core, (-1, left_rank,
                                   shape[0][core_idx], right_rank))
            if not is_batch_case:
              print(tf.shape(curr_core))
              print('curr_core',curr_core)
              print('static shape:',curr_core.get_shape())
              curr_core = tf.squeeze(curr_core, axis=0)

        tt_cores.append(curr_core)

      combined_ranks = zip(tt_left.get_tt_ranks(), right.get_tt_ranks())
      out_ranks = [a * b for a, b in combined_ranks]

  if not is_batch_case:
    return TensorTrain(tt_cores, tt_left.get_raw_shape(), out_ranks)
  else:
      return TensorTrainBatch(tt_cores, tt_left.get_raw_shape(), out_ranks,
                            batch_size=out_batch_size)



def assign(ref, value, validate_shape=None, use_locking=None, name=None):
    new_cores = []
    # if name is None:
    # name= ''
    with tf.variable_scope(name):
        for i in range(ref.ndims()):
            #curr_core_ref = ref.tt_cores[i]
            new_cores.append(tf.assign(ref.tt_cores[i], value.tt_cores[i],
                                 validate_shape=False,
                                 use_locking=use_locking))
    if isinstance(value, TensorTrainBatch):
        return TensorTrainBatch(new_cores, value.get_raw_shape(),
                            value.get_tt_ranks(), value.batch_size,
                            convert_to_tensors=False)
    else:
        return TensorTrain(new_cores, value.get_raw_shape(),
                                  value.get_tt_ranks(),
                                  convert_to_tensors=False)




#input will be original grad
def rieman_proj_first_core(grad,point):
    ndims = grad.ndims()
    dtype = grad.dtype
    #shape is 2 dimensional array
    shape = shapes.lazy_raw_shape(grad)[0]
    tt_ranks = shapes.lazy_tt_ranks(grad)
    #riemannian_grad = t3f.riemannian.project(grad,point)
    #first_tt_core = riemannian_grad.tt_cores[0]
    first_tt_core = grad.tt_cores[0]
    #normalization
    for i in range(shape[0]):
        slice = tf.expand_dims(first_tt_core[:,i,:]/tf.norm(first_tt_core[:,i,:]),1)
        if i ==0:
            curr_core = slice
        else:
            curr_core = tf.concat((curr_core,slice),axis=1)
    #print('calling first core:', curr_core)
    return curr_core

def rieman_proj_last_core(grad,point):
    ndims = grad.ndims()
    dtype = grad.dtype
    shape = shapes.lazy_raw_shape(grad)[0]
    tt_ranks = shapes.lazy_tt_ranks(grad)
    #riemannian_grad = t3f.riemannian.project(grad,point)
    #last_tt_core = riemannian_grad.tt_cores[-1]
    last_tt_core = grad.tt_cores[-1]
    #normalization
    for i in range(shape[-1]):
        slice = tf.expand_dims(last_tt_core[:,i,:]/tf.norm(last_tt_core[:,i,:]),1)
        if i ==0:
            curr_core = slice
        else:
            curr_core = tf.concat((curr_core,slice),axis=1)
    return curr_core



#input will be original grad
def skew_proj(grad,point):
    ndims = grad.ndims()
    dtype = grad.dtype
    #shape is 2 dimensional array
    shape = shapes.lazy_raw_shape(grad)[0]
    new_cores = []
    new_cores.append(rieman_proj_first_core(grad,point))
    #print(new_cores)
    tt_ranks = shapes.lazy_tt_ranks(grad)
    for core_index in range(1,ndims-1):
        for j in range(shape[core_index]):
            tt_core = grad.tt_cores[core_index]
            #print(tt_core)
            slice = tt_core[:,j,:]
            #unsqueeze in the mid dimension
            skew = tf.expand_dims((slice - tf.transpose(slice))/2,1)
            if j == 0:
                curr_core = skew
            else:
                curr_core = tf.concat((curr_core,skew),axis=1)
        new_cores.append(curr_core)
    new_cores.append(rieman_proj_last_core(grad,point))
    #print(new_cores)
    return TensorTrain(new_cores)


#for intermidiate TT cores, do cayley matplotlib
#for first and last TT cores, do riemannian update
def Cayley(ref):
    ndims = ref.ndims()
    dtype = ref.dtype
    shape = shapes.lazy_raw_shape(ref)[0]
    tt_ranks = shapes.lazy_tt_ranks(ref)
    new_cores = []
    new_cores.append(ref.tt_cores[0])
    output = []
    #internal tt cores
    for core_index in range(1,ndims-1):
        for j in range(shape[core_index]):
            tt_core = ref.tt_cores[core_index]
            slice = tt_core[:,j,:]
            identity = tf.eye(tt_ranks[core_index])
            cayley = tf.expand_dims(tf.matmul(identity - slice,tf.matrix_inverse(identity + slice)),1)
            if j ==0:
                curr_core = cayley
            else:
                curr_core = tf.concat((curr_core,cayley),axis=1)
        new_cores.append(curr_core)
    new_cores.append(ref.tt_cores[-1])
    #print(new_cores)
    #return TensorTrain(new_cores, shape,tt_ranks)
    return TensorTrain(new_cores)


#################################################################################
#OTT01
shape = 3*[10]
one_tt = t3f.tensor_ones(shape)
one_tt = t3f.get_variable('one_tt0',initializer=one_tt, trainable=False)

input_dense = np.arange(-500,500)
input_dense = input_dense.reshape(10,10,10)
#print('input',input_dense)

init_X = t3f.to_tt_tensor(input_dense.astype(np.float32),max_tt_rank=2)
print('init_X',init_X)
X = t3f.get_variable('X1',initializer=init_X)

gradF = self_multiply(X, self_multiply(X,X) - one_tt)
print('gradF:',gradF)
Riemannian_grad =  skew_proj(gradF,X)
print('riemannian_grad:',Riemannian_grad)
alpha=0.1

train_step = assign(X, Cayley(-alpha*Riemannian_grad))

sess = tf.Session()
#print('dynamic shape',sess.run(tf.shape(train_step.tt_cores[0])))
F = t3f.frobenius_norm(0.5*(self_multiply(X,X) - one_tt))

sess.run(tf.global_variables_initializer())

log_r_01=[]
print('OTT01 begin!')
start_time = time.time()
for i in range(5):
    F_v,_ = sess.run([F,train_step.op])
    if i%1==0:
        print(i,F_v)
    log_r_01.append(F_v)
print('OTT01 time:',time.time()-start_time)
print(sess.run(t3f.full(X)))
##################################################################################
#for i in range)(5):


#####################################################################################
#Riemannian optimization
shape = 3*[10]
one_tt = t3f.tensor_ones(shape)
one_tt = t3f.get_variable('one_tt1',initializer=one_tt, trainable=False)

input_dense = np.arange(-500,500)
input_dense = input_dense.reshape(10,10,10)
print('input',input_dense)

init_X = t3f.to_tt_tensor(input_dense.astype(np.float32),max_tt_rank=16)
X = t3f.get_variable('X1',initializer=init_X)

gradF = X*(X*X - one_tt)
#gradF = X
#gradF = X-one_tt
riemannian_grad = t3f.riemannian.project(gradF,X)

alpha=0.1
train_step = t3f.assign(X, t3f.round(X - alpha*riemannian_grad, max_tt_rank=16))
#loss value
#F = 0.25*t3f.frobenius_norm_squared(X*X - one_tt)
F = t3f.frobenius_norm(0.5*(X*X - one_tt))
sess = tf.Session()
sess.run(tf.global_variables_initializer())

log_r_01=[]
print('rieman01 begin!')
start_time = time.time()
for i in range(80):
    F_v,_ = sess.run([F,train_step.op])
    if i%1==0:
        print(i,F_v)
    log_r_01.append(F_v)
print('riemann01 time:',time.time()-start_time)
print(sess.run(t3f.full(X)))


#######################################################################
#Adam lr=0.01
shape = (10,10,10)
input_dense = np.arange(-500,500)
#input_dense[9] = 1
input_dense = input_dense.reshape(10,10,10)
#print('input',input_dense)

input_tt = t3f.to_tt_tensor(input_dense.astype(np.float32), max_tt_rank = 16)
print('one_tt',input_tt.tt_ranks())
input_tt = t3f.get_variable('init2', initializer=input_tt)

learning_rate_adam=1e-2

one_tt = t3f.tensor_ones(shape)
one_tt = t3f.get_variable('one_tt2',initializer=one_tt,trainable=False)
#without rounding
#gradF = input_tt*(input_tt*input_tt - one_tt)
#gradF = input_tt - one_tt
#with rounding
eps=1e-6
max_ranks=8
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_gd)
#train_step = t3f.assign(input_tt, t3f.round(input_tt - learning_rate_gd*gradF, max_tt_rank=16))
f_norm = 0.25 * t3f.frobenius_norm_squared(input_tt*input_tt - one_tt, differentiable=True,name='frobenius_norm')

optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate_adam)
adam_step = optimizer2.minimize(f_norm)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

log_a_001 = []
print('adam001 begin!')
for i in range(300):
    _, tr_fnorm_v = sess.run([adam_step, f_norm])
    log_a_001.append(tr_fnorm_v)
    if i % 10 == 0:
        print(i, tr_fnorm_v)

################################################################3


#Adam lr=0.001
shape = (10,10,10)
input_dense = np.arange(-500,500)
input_dense = input_dense.reshape(10,10,10)
#print('input',input_dense)

input_tt = t3f.to_tt_tensor(input_dense.astype(np.float32), max_tt_rank = 16)
input_tt = t3f.get_variable('init3', initializer=input_tt)

learning_rate_adam=1e-3

one_tt = t3f.tensor_ones(shape)
one_tt = t3f.get_variable('one_tt3',initializer=one_tt,trainable=False)
#without rounding
#gradF = input_tt*(input_tt*input_tt - one_tt)
#gradF = input_tt - one_tt
#with rounding
eps=1e-6
max_ranks=8
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_gd)
#train_step = t3f.assign(input_tt, t3f.round(input_tt - learning_rate_adam*gradF, max_tt_rank=16))
f_norm = 0.25 * t3f.frobenius_norm_squared(input_tt*input_tt - one_tt, differentiable=True,name='frobenius_norm')

optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate_adam)
adam_step = optimizer2.minimize(f_norm)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

log_a_0001 = []
print('adam0001 begin!')
for i in range(10000):
    _, tr_fnorm_v = sess.run([adam_step, f_norm])
    log_a_0001.append(tr_fnorm_v)
    if i % 10 == 0:
        print(i, tr_fnorm_v)

print('adam output',sess.run(t3f.full(input_tt)))
#####################################################################
#Adam lr=0.1

#Adam lr=0.001
shape = (10,10,10)
input_dense = np.arange(-500,500)
#input_dense[9] = 1
input_dense = input_dense.reshape(10,10,10)
#print('input',input_dense)

input_tt = t3f.to_tt_tensor(input_dense.astype(np.float32), max_tt_rank = 16)
input_tt = t3f.get_variable('init30', initializer=input_tt)

learning_rate_gd=1e-2
learning_rate_adam=0.1

one_tt = t3f.tensor_ones(shape)
one_tt = t3f.get_variable('one_tt30',initializer=one_tt,trainable=False)
#without rounding
#gradF = input_tt*(input_tt*input_tt - one_tt)
#gradF = input_tt - one_tt
#with rounding
eps=1e-6
max_ranks=8
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_gd)
#train_step = t3f.assign(input_tt, t3f.round(input_tt - learning_rate_gd*gradF, max_tt_rank=16))
f_norm = 0.25 * t3f.frobenius_norm_squared(input_tt*input_tt - one_tt, differentiable=True,name='frobenius_norm')

optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate_adam)
adam_step = optimizer2.minimize(f_norm)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

log_a_01 = []
print('adam0001 begin!')
for i in range(10000):
    _, tr_fnorm_v = sess.run([adam_step, f_norm])
    log_a_01.append(tr_fnorm_v)
    if i % 10 == 0:
        print(i, tr_fnorm_v)
print('adam output',sess.run(t3f.full(input_tt)))
#######################################################################
#GD lr=0.1
shape = (10,10,10)
input_dense = np.arange(-500,500)
input_dense = input_dense.reshape(10,10,10)
#print('input',input_dense)

input_tt = t3f.to_tt_tensor(input_dense.astype(np.float32), max_tt_rank = 16)
input_tt = t3f.get_variable('init4', initializer=input_tt)

learning_rate_gd=1e-1

one_tt = t3f.tensor_ones(shape)
one_tt = t3f.get_variable('one_tt4',initializer=one_tt,trainable=False)
#without rounding
gradF = input_tt*(input_tt*input_tt - one_tt)
#gradF = input_tt - one_tt
#with rounding
eps=1e-6
max_ranks=8
gd_step = t3f.assign(input_tt, t3f.round(input_tt - learning_rate_gd*gradF, max_tt_rank=16))
f_norm = 0.25 * t3f.frobenius_norm_squared(input_tt * input_tt-one_tt, differentiable=True, name='frobenius_norm')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

log_g_01= []
print('gd01 begin!')
start_time = time.time()
for i in range(100):
    _, tr_fnorm_v = sess.run([gd_step.op, f_norm])
    log_g_01.append(tr_fnorm_v)
    if i % 10 == 0:
        print(i, tr_fnorm_v)
print('gd01 time:',time.time()-start_time)
print(sess.run(t3f.full(input_tt)))
#####################################################################
#GD lr=0.01
shape = (10,10,10)
input_dense = np.arange(-500,500)
#input_dense[9] = 1
input_dense = input_dense.reshape(10,10,10)
#print('input',input_dense)

input_tt = t3f.to_tt_tensor(input_dense.astype(np.float32), max_tt_rank = 16)
input_tt = t3f.get_variable('init5', initializer=input_tt)

learning_rate_gd=1e-2

one_tt = t3f.tensor_ones(shape)
one_tt = t3f.get_variable('one_tt5',initializer=one_tt,trainable=False)
#without rounding
gradF = input_tt*(input_tt*input_tt - one_tt)
#gradF = input_tt - one_tt
#with rounding
eps=1e-6
max_ranks=8
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_gd)
gd_step = t3f.assign(input_tt, t3f.round(input_tt - learning_rate_gd*gradF, max_tt_rank=16))
f_norm = 0.25 * t3f.frobenius_norm_squared(input_tt*input_tt - one_tt, differentiable=True,name='frobenius_norm')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

log_g_001 = []
print('gd001begin!')
for i in range(300):
    _, tr_fnorm_v = sess.run([gd_step.op, f_norm])
    log_g_001.append(tr_fnorm_v)
    if i % 10 == 0:
        print(i, tr_fnorm_v)

######################################################################
#GD lr=0.001
shape = (10,10,10)
input_dense = np.arange(-500,500)
input_dense = input_dense.reshape(10,10,10)
#print('input',input_dense)

input_tt = t3f.to_tt_tensor(input_dense.astype(np.float32), max_tt_rank = 16)
input_tt = t3f.get_variable('init6', initializer=input_tt)

learning_rate_gd=1e-3

one_tt = t3f.tensor_ones(shape)
one_tt = t3f.get_variable('one_tt6',initializer=one_tt,trainable=False)
#without rounding
gradF = input_tt*(input_tt*input_tt - one_tt)
#gradF = input_tt - one_tt
#with rounding
eps=1e-6
max_ranks=8
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_gd)
gd_step = t3f.assign(input_tt, t3f.round(input_tt - learning_rate_gd*gradF, max_tt_rank=16))
f_norm = 0.25 * t3f.frobenius_norm_squared(input_tt*input_tt - one_tt, differentiable=True,name='frobenius_norm')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

log_g_0001 = []
print('gd0001 begin!')
for i in range(300):
    _, tr_fnorm_v = sess.run([gd_step.op, f_norm])
    log_g_0001.append(tr_fnorm_v)
    if i % 10 == 0:
        print(i, tr_fnorm_v)
########################################################################
#lr decay GD
shape = (10,10,10)
input_dense = np.arange(-500,500)
#input_dense[9] = 1
input_dense = input_dense.reshape(10,10,10)
#print('input',input_dense)

input_tt = t3f.to_tt_tensor(input_dense.astype(np.float32), max_tt_rank = 16)
input_tt = t3f.get_variable('init_decay', initializer=input_tt)

learning_rate_gd=0.1

one_tt = t3f.tensor_ones(shape)
one_tt = t3f.get_variable('one_tt_decay',initializer=one_tt,trainable=False)
#without rounding
gradF = input_tt*(input_tt*input_tt - one_tt)
#gradF = input_tt - one_tt
#with rounding
eps=1e-6
max_ranks=8
gd_step_01 = t3f.assign(input_tt, t3f.round(input_tt - learning_rate_gd*gradF, max_tt_rank=16))
gd_step_001 = t3f.assign(input_tt, t3f.round(input_tt - 0.5*learning_rate_gd*gradF, max_tt_rank=16))
gd_step_0001 = t3f.assign(input_tt, t3f.round(input_tt - 0.001*learning_rate_gd*gradF, max_tt_rank=16))

f_norm = 0.25 * t3f.frobenius_norm_squared(input_tt*input_tt - one_tt, differentiable=True,name='frobenius_norm')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

log_g_01_1 = []
log_g_001_2=[]
log_g_0001_3=[]
print('gd lr decay begin!')
for i in range(120):
    _, tr_fnorm_v = sess.run([gd_step_01.op, f_norm])
    log_g_01_1.append(tr_fnorm_v)
    if i % 10 == 0:
        print(i, tr_fnorm_v)

for i in range(100):
    _, tr_fnorm_v = sess.run([gd_step_001.op, f_norm])
    log_g_001_2.append(tr_fnorm_v)
    if i % 10 == 0:
        print(i+120, tr_fnorm_v)

for i in range(80):
    _, tr_fnorm_v = sess.run([gd_step_0001.op, f_norm])
    log_g_0001_3.append(tr_fnorm_v)
    if i % 10 == 0:
        print(i+220, tr_fnorm_v)

log_g_decay=log_g_01_1+log_g_001_2+log_g_0001_3

########################################################################
#plot
x = np.arange(0,300)
fig = plt.figure()

ax1,ax2 = fig.subplots(1,2,sharex=True,sharey=True)
ax1.set_yscale('log')
ax2.set_yscale('log')

ax1.plot(x,log_r_01,label='lr01Rieman')
ax1.plot(x,log_g_01,label='lr01GD')
ax1.plot(x,log_g_001,label='lr001GD')
ax1.plot(x,log_g_0001,label='lr0001GD')
ax1.plot(x,log_g_decay,label='lrdecayGD')
#ax1.plot(x,log_g_01_nr,label='lr-1GDnr')

ax2.plot(x,log_r_01,label='lr01Rieman')
ax2.plot(x,log_a_01,label='lr01Adam')
ax2.plot(x,log_a_001,label='lr001Adam')
ax2.plot(x,log_a_0001,label='lr0001Adam')

ax1.set_xlabel('Iteration')
ax1.set_ylabel('frobenious_loss')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('frobenious_loss')

ax1.set_title('Reimannian vs GD')
ax2.set_title('Reimannian vs Adam')
ax1.legend(loc='best')
ax2.legend(loc='best')
plt.show()
