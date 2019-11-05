#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: GeoMAN
@time: 2019/11/4 21:08
@desc:
"""

import tensorflow as tf


class LSTMCell:
    def __init__(self, num_units, activation=None, initializer=None, forget_bias=0.0):
        self._num_units = num_units
        self._activation = activation
        self._initializer = initializer
        self._forget_bias = forget_bias

        self.built = False

    def build(self, input_shape):
        last_shape = input_shape[-1]
        with tf.variable_scope('LSTMCell'):
            self._kernel = tf.get_variable('kernel', shape=[last_shape + self._num_units, 4 * self._num_units],
                                           dtype=tf.float32,
                                           initializer=self._initializer)
            self._bias = tf.get_variable('bias', shape=[4 * self._num_units],
                                         dtype=tf.float32,
                                         initializer=tf.constant_initializer(0.0))

        self.built = True

    def call(self, inputs, states):
        if self.built is False:
            self.build(inputs.get_shape().as_list())

        c_prev, m_prev = states

        # TODO infer rank
        lstm_matrix = tensordot(tf.concat([inputs, m_prev], axis=-1),
                                self._kernel)
        # shape -> [..., 4 * num_units]
        lstm_matrix = lstm_matrix + self._bias
        # shape -> [..., num_units]
        i, j, f, o = tf.split(lstm_matrix, 4, axis=-1)

        c = (tf.nn.sigmoid(f + self._forget_bias) * c_prev +
             tf.sigmoid(i) * self._activation(j))
        m = tf.nn.sigmoid(o) * self._activation(c)

        return m, [c, m]


def tensordot(a, b):
    last_idx_a = len(a.get_shape().as_list()) - 1
    return tf.tensordot(a, b, [[last_idx_a], [0]])
