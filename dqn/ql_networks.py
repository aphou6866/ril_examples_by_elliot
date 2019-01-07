import os, sys, json
import numpy as np
import tensorflow as tf


# import tensorflow as tf

# from ql_networks import build_dense_network, build_dense_duel

# X, Y, params= build_dense_network( 5, 3, 2, 7, 'test_network')

# X, Y, params= build_dense_duel( 5, 3, 2, 7, 'test_network')

# g= tf.get_default_graph()

# tf.summary.FileWriter('logs',g).close()



def build_dense_network( n, m, layers, hiddens, scope):

    with tf.variable_scope(scope):
        X= tf.placeholder( tf.float32, [None, n], name='X')
        for i in range( layers):
            #w_init = tf.random_normal_initializer(0., .1)
            if i==0:
                sz= n
                net= X
            else:
                sz= hiddens
                
            w= tf.Variable( tf.truncated_normal( [ sz, hiddens ], name='init%02dw'%i), name='hidden%02dw'%i)
            b= tf.Variable( tf.constant(0.0, shape=[hiddens]), name='hidden%02db'%i )
            net= tf.nn.relu( tf.matmul( net, w) +b )
            
        w= tf.Variable( tf.truncated_normal( [ hiddens, m ],name='initw'), name='netoutw')
        b= tf.Variable( tf.constant(0.0, shape=[m]), name='netoutb')
        Y = tf.matmul( net, w) +b
        # Do not use softmax for pendulum that increasing unstability.
        #out= tf.nn.relu( tf.matmul( net, w) +b )
        #Y= pred_softmax = tf.nn.softmax( out, name='softmax_out')
    params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

    return X, Y, params


def build_dense_duel( n, m, layers, hiddens, scope):

    with tf.variable_scope(scope):
        X= tf.placeholder( tf.float32, [None, n], name='X')
        for i in range( layers):
            #w_init = tf.random_normal_initializer(0., .1)
            if i==0:
                sz= n
                net= X
            else:
                sz= hiddens
                
            w= tf.Variable( tf.truncated_normal( [ sz, hiddens ], name='init%02dw'%i), name='hidden%02dw'%i)
            b= tf.Variable( tf.constant(0.0, shape=[hiddens]), name='hidden%02db'%i )
            net= tf.nn.relu( tf.matmul( net, w) +b )
            
        w= tf.Variable( tf.truncated_normal( [ hiddens, 1 ],name='initw'), name='valueW')
        b= tf.Variable( tf.constant(0.0, shape=[1]), name='valueB')
        V = tf.matmul( net, w) +b
        
        w= tf.Variable( tf.truncated_normal( [ hiddens, m ],name='initw'), name='advantageW')
        b= tf.Variable( tf.constant(0.0, shape=[m]), name='advantageB')
        A = tf.matmul( net, w) +b
        
        # Q = V(s) + A(s,a)
        dA= (A - tf.reduce_mean( A, axis=1, keepdims=True))
        Y = V + dA
        
    params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

    return X, Y, params

    
    
