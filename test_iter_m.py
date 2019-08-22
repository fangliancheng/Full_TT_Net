import numpy as np
import timeit
#import torch
#import tntorch as tn
from ops_ex_maxnorm import *

#tf.enable_eager_execution()

shape = (30,30,10)
# test_dense = np.arange(-500,500)
# print('test_dense', test_dense[500])
# test_dense[500] = 1
# print('new value', test_dense[500])
# test_dense = test_dense.reshape(10,10,10)

test_dense = np.arange(-4500,4500)
test_dense[0] = 0
test_dense[5] = -140
print('test_dense', test_dense[9])
test_dense[9] = 1
print('new value', test_dense[9])
test_dense = test_dense.reshape(30,30,10)
print('dense', test_dense)


test_tt = t3f.to_tt_tensor(test_dense.astype(np.float32), max_tt_rank = 4)
test_tt = t3f.get_variable('uuu', initializer = test_tt, trainable = False)
one_tt = t3f.tensor_ones(shape)
one_tt = t3f.get_variable('one', initializer = one_tt, trainable = False)
#print('one_tt',one_tt)
err = 1e-3
u_init = t3f.random_tensor(shape,tt_rank = 4)
u_init = t3f.get_variable('u_0', initializer = u_init, trainable = True)
#loss = t3f.frobenius_norm(one_tt - t3f.multiply(test_tt,u_init))

#TODO:why t3f.full can take derivative, but other cannot?
f_norm = tf.reduce_max((t3f.full(one_tt)-t3f.full(test_tt)*t3f.full(u_init))**2)
m_norm = max_norm(u_init,shape)
optimizer = tf.train.AdamOptimizer(learning_rate = 1e-2)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
step = optimizer.minimize(f_norm)
sess = tf.Session()

# print('global_variables:', tf.global_variables())
# print('local_variables:', tf.local_variables())
# print('train_variables:', tf.trainable_variables())
# print('model_variables:', tf.model_variables())

sess.run(tf.global_variables_initializer())

train_fnorm_hist = []
train_mnorm_hist = []
for i in range(500):
    _, tr_fnorm_v, tr_mnorm_v = sess.run([step, f_norm, m_norm])
    train_fnorm_hist.append(tr_fnorm_v)
    train_mnorm_hist.append(tr_mnorm_v)
    if i % 100 == 0:
        print(i, tr_fnorm_v, tr_mnorm_v)

#u_init: A Tensor Train variable of shape (10, 10, 10), TT-ranks: (1, 4, 4, 1)
#print('u_init:', u_init)

start000 = timeit.default_timer()
test_cha = characteristic(test_tt,shape,err,u_init)
print(test_cha)
#must sess.run(full(tt)) but not sess.run(tt)
test_cha = sess.run(t3f.full(test_cha))
print('test_cha:',test_cha)
stop000 = timeit.default_timer()
print('test_cha time:', stop000-start000)

start00 = timeit.default_timer()
test_maxnorm = max_norm(test_tt,shape, err)
test_maxnorm = sess.run(test_maxnorm)
print('test_maxnorm:', test_maxnorm)
stop00 = timeit.default_timer()
print('max_norm time', stop00 - start00)

start0 = timeit.default_timer()
test_point_inv = point_inv(test_tt,shape,u_init)
test_point_inv = sess.run(t3f.full(test_point_inv))
print('test_inv', test_point_inv)
stop0 = timeit.default_timer()
print('time_point_inv', stop0-start0)


start1 = timeit.default_timer()
test_sign = sign(test_tt, shape, u_init, err)
test_sign = sess.run(t3f.full(test_sign))
print('test_sign',test_sign)
stop1 = timeit.default_timer()
print('time_sign', stop1-start1)
#ugly, u_init is a augument for point_inv
start2 = timeit.default_timer()
test_relu = level_set(test_tt, shape, err, u_init)
relu_test = sess.run(t3f.full(test_relu))
stop2 = timeit.default_timer()
sess.close()
#print(type(relu_test))
print('relu_test',relu_test)
print('time_relu:',stop2-start2)
