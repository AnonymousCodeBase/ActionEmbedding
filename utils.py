import numpy as np
import tensorflow as tf


EPS = 1e-6


def mlp(x, hiddens, activation=None, reguliazer=None):
    for h in hiddens:
        x = tf.layers.dense(x, h, activation=activation, kernel_regularizer=reguliazer)
    return x


def mlp1(x, hiddens, activation=None, output_activation=None, reguliazer=None):
    for h in hiddens[:-1]:
        x = tf.layers.dense(x, h, activation=activation, kernel_regularizer=reguliazer)
    x = tf.layers.dense(x, hiddens[-1], activation=output_activation, kernel_regularizer=reguliazer)
    return x


def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)


def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

