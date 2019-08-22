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
import torch
import t3nsor as t3

# t = tn.randn(128,128,ranks_tt=10,requires_grad=True)
# t
# print(t)
#
# def loss(t):
#     #take left upper part
#     return tn.norm(t[:64, :64])
#
# tn.optimize(t,loss)
# print(t)
# plt.imshow(t.numpy(),cmap='gray')
# plt.show()
#
# b = tn.randn(128,128,ranks_tt=10,requires_grad=True)
# print("111")
# print("111",t*b)
#
# a = t3f.random_tensor((3,3,3),tt_rank =2)
# print(a.dtype)
# b = t3f.frobenius_norm(a)
# b_val = 2
# print(b.dtype)
# bb = tf.cast(b, tf.float64)
# print(type(bb))
#
# x = tf.constant([1.8, 2.2], dtype=tf.float32)
# xx = tf.cast(x, tf.int32)  # [1, 2], dtype=tf.int32
# print(x.dtype)
# print(type(x))
# print(type(xx))
#
# # print(1)
# # #test inner function parameter access
# # def outer(x):
# #     x = x+1
# #     def inner():
# #         y = x + 5
# #         x = x + 5
# #         return [x,y]
# #     [x,y] = inner()
# #     print(in)
# #     return [x, y]
# #
# # a = 2
# # [b,c] = outer(a)
# # print(b)
# # print(c)
#
# a=tf.Variable(tf.constant([0,1,2],dtype=tf.int32))
# b=tf.Variable(tf.constant([1,1,1],dtype=tf.int32))
# recall=tf.metrics.recall(b,a)
# print('local_variables',tf.local_variables())
# init1=tf.global_variables_initializer()
# init2=tf.local_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init1)
#     sess.run(init2)
#     rec=sess.run(recall)
#     print(rec)
#from tensorflow.python.ops.linalg import linear_operator_full_matrix,linear_operator_kronecker

def tf_kron(a,b):
    a_shape = [a.shape[0].value,a.shape[1].value]
    b_shape = [b.shape[0].value,b.shape[1].value]
    return tf.reshape(tf.reshape(a,[a_shape[0],1,a_shape[1],1])*tf.reshape(b,[1,b_shape[0],1,b_shape[1]]),[a_shape[0]*b_shape[0],a_shape[1]*b_shape[1]])


def self_multiply(tt_left, tt_right):
    ndims = tt_left.ndims()
    shape = shapes.lazy_raw_shape(tt_left)[0]
    #tt_ranks = shapes.lazy_tt_ranks(tt_left)
    new_cores = []
    for core_idx in range(ndims):
        for j in range(shape[core_idx]):
            left_curr_core = tt_left.tt_cores[core_idx]
            right_curr_core = tt_right.tt_cores[core_idx]
            left_slice = left_curr_core[:,j,:]
            right_slice = right_curr_core[:,j,:]
            print('left_slice:',left_slice)
            print('right slice', right_slice)
            #operator1 = tf.linalg.LinearOperatorFullMatrix(left_slice)
            #operator2 = tf.linalg.LinearOperatorFullMatrix(right_slice)
            #operator = tf.linalg.LinearoperatorKronecker([operator1,operator2])
            out_slice = tf.expand_dims(tf_kron(left_slice,right_slice),1)
            print('out_slice',out_slice)
            #out_slice = tf.expand_dims(tf.contrib.kfac.utils.kronecker_product(left,slice,right_slice),1)
            if j == 0:
                out_core = out_slice
            else:
                out_core = tf.concat((out_core,out_slice),axis=1)
        print('out_core！！！！！',out_core)
        new_cores.append(out_core)
    return TensorTrain(new_cores)


def test_1():
    a = tf.Variable([10, 20])
    b = tf.assign(a, [20, 30,40],validate_shape=False)
    #c = a + [10, 20, 10]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("test_1 run a : ",sess.run(a)) # => [10 20]
        #print("test_1 run c : ",sess.run(c)) # => [10 20]+[10 20] = [20 40] 因为b没有被run所以a还是[10 20]
        print("test_1 run b : ",sess.run(b)) # => ref:a = [20 30] 运行b，对a进行assign
        print("test_1 run a again : ",sess.run(a)) # => [20 30] 因为b被run过了，所以a为[20 30]
        print("test_1 run c again : ",sess.run(c)) # => [20 30] + [10 20] = [30 50] 因为b被run过了，所以a为[20,30], 那么c就是[30 50]


def test_2():
    a = tf.Variable([10, 20])
    b = tf.assign(a, [20, 30])
    c = b + [10, 20]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(a)) # => [10 20]
        print(sess.run(c)) # => [30 50] 运行c的时候，由于c中含有b，所以b也被运行了
        print(sess.run(a)) # => [20 30]


if __name__ =='__main__':
    # eps = 1e-16
    # a = t3f.tensor_ones((3,3,3))
    # b = a
    # c_truth = t3f.multiply(b,b)
    # c_self = self_multiply(b,b)
    # with tf.Session() as sess:
    #     dense1 = sess.run(t3f.full(c_truth))
    #     dense2 = sess.run(t3f.full(c_self))
    # print('dense1',dense1)
    # print('dense2',dense2)
    shape = 3*[10]
    one_tt = t3f.tensor_ones(shape)

    #test_1()
    #test_2()
