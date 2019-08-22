import tensorflow as tf
import t3f
import numpy as np
#import torch
#import tntorch as tn
from ops_ex_fnorm import *

#tf.enable_eager_execution()

shape = (10,10,10)
#test_dense = np.random.randn(10,10,10)
test_dense = np.arange(-500,500).reshape(10,10,10)
test_tt = t3f.to_tt_tensor(test_dense.astype(np.float32), max_tt_rank = 4)
test_tt = t3f.get_variable('uuu', initializer = test_tt, trainable = False)
one_tt = t3f.tensor_ones(shape)
one_tt = t3f.get_variable('one', initializer = one_tt, trainable = False)
print('one_tt',one_tt)
err = 0.01
u_init = t3f.random_tensor(shape,tt_rank = 4)
u_init = t3f.get_variable('u_0', initializer = u_init, trainable = True)
#loss = t3f.frobenius_norm(one_tt - t3f.multiply(test_tt,u_init))

#TODO:why t3f.full can take derivative, but other cannot?
loss = tf.reduce_sum((t3f.full(one_tt)-t3f.full(test_tt)*t3f.full(u_init))**2)
print(loss)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.01)
step = optimizer.minimize(loss)
sess = tf.Session()

print('global_variables:', tf.global_variables())
print('local_variables:', tf.local_variables())
print('train_variables:', tf.trainable_variables())
print('model_variables:', tf.model_variables())

sess.run(tf.global_variables_initializer())

train_loss_hist = []
for i in range(50000):
    _, tr_loss_v = sess.run([step, loss])
    train_loss_hist.append(tr_loss_v)
    if i % 100 == 0:
        print(i, tr_loss_v)

#u_init: A Tensor Train variable of shape (10, 10, 10), TT-ranks: (1, 4, 4, 1)
print('u_init:', u_init)

#ugly, u_init is a augument for point_inv
test_relu = level_set(test_tt, shape, err, u_init)
relu_test = sess.run(t3f.full(test_relu))
sess.close()
print(type(relu_test))
print(relu_test)
