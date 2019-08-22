import tensorflow as tf
import t3f
import timeit
from ops_ex_maxnorm import *
import matplotlib.pyplot as plt

shape = (10,10,10)
input_dense = np.arange(-500,500)
#input_dense[9] = 1
input_dense = input_dense.reshape(10,10,10)
#print('input',input_dense)

input_tt = t3f.to_tt_tensor(input_dense.astype(np.float32), max_tt_rank = 16)
input_tt = t3f.get_variable('init', initializer=input_tt)

learning_rate_gd=1e-2
learning_rate_adam=1e-2

one_tt = t3f.tensor_ones(shape)
one_tt = t3f.get_variable('one_tt',initializer=one_tt,trainable=False)
#without rounding
gradF = input_tt*(input_tt*input_tt - one_tt)
gradF = input_tt - one_tt
#with rounding
eps=1e-6
max_ranks=8
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_gd)
train_step = t3f.assign(input_tt, t3f.round(input_tt - learning_rate_gd*gradF, max_tt_rank=16))
f_norm = 0.5 * t3f.frobenius_norm_squared(input_tt - one_tt, differentiable=True,name='frobenius_norm')

optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate_adam)
adam_step = optimizer2.minimize(f_norm)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

train_fnorm_hist = []
start = timeit.default_timer()
for i in range(300):
    _, tr_fnorm_v = sess.run([adam_step, f_norm])
    train_fnorm_hist.append(tr_fnorm_v)
    #train_mnorm_hist.append(tr_mnorm_v)
    if i % 10 == 0:
        print(i, tr_fnorm_v)

stop = timeit.default_timer()
plt.plot(train_fnorm_hist)
#plt.show()
print(train_fnorm_hist)
print('optim_time:', stop-start)
#print(sess.run(t3f.full(input_tt)))
