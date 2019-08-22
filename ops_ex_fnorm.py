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

#Addition
#Hadmard product
#Scalar product
#F norm

# #TODO:opt
# def opt(tt_tensor, shape):
#     #ramdom initializer
#     initializer = t3f.random_tensor(shape,tt_rank = 4)
#     u_0 = t3f.get_variable('u_tt', initializer=initializer)
#     one_tt = t3f.tensor_ones(shape)
#     obj = t3f.frobenius_norm(one_tt - t3f.multiply(tt_tensor, u_0))
#
#     optimizer = tf.train.GradientDescentOptimizer(0.05)
#     train = optimizer.minimize(obj)
#
#     init = tf.initialize_all_variables()
#
#     with tf.Session() as session:
#         session.run(initializer)
#         print("starting at","u_0:", sess.run(u_0), "object:", session.run(obj))
#         for step in range(100):
#             session.run(train)
#             print("step", step, "u_0", session.run(u_0), "object", session.run(obj))
#
#     return u_0


#TODO:Pointwise inverse
#Args: tt_tensor:
#      shape:
#      err: rounding error
def point_inv(tt_tensor,shape, err, point_inv_aug):
    # Choose u_tuple_init s.t condition holds
    #TODO:write opt
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
        return tf.less(err*t3f.frobenius_norm(tt_tensor), t3f.frobenius_norm(one_tt - t3f.multiply(tt_tensor,u_tt)))

    def body(u_tt_cores1,u_tt_cores2,u_tt_cores3):
        u_tt = TensorTrain([u_tt_cores1,u_tt_cores2,u_tt_cores3])
        return tuple(t3f.round(t3f.multiply(u_tt, 2*one_tt-t3f.multiply(tt_tensor, u_tt)), 4).tt_cores)

    #c = lambda (u_tt,): tf.less(err*t3f.frobenius_norm(tt_tensor), t3f.frobenius_norm(one_tt - t3f.multiply(tt_tensor,u_tt)))
    #b = lambda (u_tt,): (t3f.round(t3f.multiply(u_tt, 2*one_tt-t3f.multiply(tt_tensor, u_tt))),)
    u_tuple_final = tf.while_loop(cond,body,u_tuple_init)
    return TensorTrain(u_tuple_final)


#TODO:sign
def sign(tt_tensor, shape, err, point_inv_aug):
    one_tt = t3f.tensor_ones(shape)
    #extra = tf.constant(1)
    #TODO: tt_tensor is a list of tf.tensor, how to pass a tuple of tf.tensor to tf.while_loop
    u_tuple_init = tuple(tt_tensor.tt_cores)
    print("u_tuple_init1 :", u_tuple_init)
    print("type:",type(u_tuple_init))
    print("core1:",u_tuple_init[0])
    #Args:u_tt_cores is tuple of tf.tensor
    #let num = len(tt_tensor.tt_cores)
    #TODO:ugly
    def cond(u_tt_cores1, u_tt_cores2, u_tt_cores3):
        #TODO:use u_tt_cores to reconstuct tt_tensor
        #TensorTrain class should be able to infer the shape from tt_cores
        u_tt = TensorTrain([u_tt_cores1, u_tt_cores2, u_tt_cores3])
        print("cond",u_tt)
        return tf.less(err*t3f.frobenius_norm(tt_tensor), t3f.frobenius_norm(one_tt - t3f.multiply(u_tt,u_tt)))

    def body(u_tt_cores1, u_tt_cores2, u_tt_cores3):
        u_tt = TensorTrain([u_tt_cores1, u_tt_cores2, u_tt_cores3])
        print("body",u_tt)
        return tuple(tf.cond(tf.less(t3f.frobenius_norm(one_tt - t3f.multiply(u_tt,u_tt)),tf.constant(1,dtype = tf.float32)),
                      lambda: t3f.round(1/2 * t3f.multiply(u_tt, 3*one_tt - t3f.multiply(u_tt, u_tt)),4).tt_cores,
                      lambda: t3f.round(1/2 * (u_tt + point_inv(u_tt, shape, err, point_inv_aug)), 4).tt_cores ))

    u_tuple_final = tf.while_loop(cond,body,u_tuple_init)
    #c = lambda i: tf.less(err*t3f.frobenius_norm(tt_tensor), t3f.frobenius_norm(one_tt - t3f.multiply(u_tt,u_tt)))
    #b = lambda i: tf.cond(tf.less(t3f.frobenius_norm(one_tt - t3f.multiply(u_tt,u_tt)),1), lambda:t3f.round(1/2 * t3f.multiply(u_tt, 3*one_tt - t3f.multiply(u_tt, u_tt))), lambda: t3f.round(1/2 * (u_tt + point_inv(u_tt, shape, err))) )
    return TensorTrain(u_tuple_final)


#TODO: characteristic
def characteristic(tt_tensor, shape, err,point_inv_aug):
    #may need careful check
    one_tt = t3f.tensor_ones(shape)
    return 1/2*(one_tt - sign(-1 * tt_tensor, shape, err, point_inv_aug))

#TODO:level set
def level_set(tt_tensor, shape, err, point_inv_aug):
    return t3f.multiply(characteristic(tt_tensor, shape, err,point_inv_aug),tt_tensor)
