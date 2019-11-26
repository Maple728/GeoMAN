#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: GeoMAN
@time: 2019/11/2 16:38
@desc:
"""

import tensorflow as tf

from models.utils import LSTMCell, tensordot


class GeoMAN:
    def __init__(self, config):
        # assign the parameters
        self.num_seqs = config['num_seqs']
        self.T = config['T']
        self.horizon = config['horizon']
        self.x_dim = config['x_dim']
        self.hidden_size = config['hidden_size']

    def train(self, sess, batch_data, lr):
        fd = {self.dropout_rate: 1.0,
              self.lr: lr,
              self.x_ph: batch_data[0],
              self.y_ph: batch_data[1],
              self.label_ph: batch_data[2]}
        _, loss, pred, real = sess.run([self.train_op, self.loss, self.y_pred, self.label_ph], feed_dict=fd)
        return loss, pred, real

    def predict(self, sess, batch_data, lr=None):
        fd = {self.dropout_rate: 0.0,
              self.lr: lr,
              self.x_ph: batch_data[0],
              self.y_ph: batch_data[1],
              self.label_ph: batch_data[2]}
        loss, pred, real = sess.run([self.loss, self.y_pred, self.label_ph], feed_dict=fd)
        return loss, pred, real

    @staticmethod
    def activate_func(x_input):
        return tf.nn.relu(x_input)

    @staticmethod
    def get_rnn_cell(rnn_hid, keep_prob):
        rnn = LSTMCell(rnn_hid, activation=tf.nn.tanh, initializer=tf.initializers.orthogonal(),
                       forget_bias=0.1)
        # rnn = tf.nn.rnn_cell.LSTMCell(rnn_hid, initializer=tf.initializers.orthogonal(),
        #                               activation=tf.nn.tanh, forget_bias=0.0)
        # rnn = tf.nn.rnn_cell.DropoutWrapper(rnn, output_keep_prob=keep_prob)
        return rnn

    @staticmethod
    def get_weights(name, shape, is_reg=True, collections=None):
        if is_reg:
            weights = tf.get_variable(name, shape=shape, dtype=tf.float32,
                                      initializer=tf.glorot_normal_initializer(),
                                      collections=collections)
        else:
            weights = tf.get_variable(name, shape=shape, dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      collections=collections)
        return weights

    @staticmethod
    def get_bias(name, shape, collections=None):
        bias = tf.get_variable(name, shape=shape, dtype=tf.float32, initializer=tf.constant_initializer(0.1),
                               collections=collections)
        return bias

    def post_build(self):
        # placeholder for hyperparameters
        self.dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        # placeholder for datas
        self.x_ph = tf.placeholder(tf.float32, shape=[None, self.num_seqs, self.T, self.x_dim],
                                   name='x')
        self.y_ph = tf.placeholder(tf.float32, shape=[None, self.num_seqs, self.T],
                                   name='y')
        self.label_ph = tf.placeholder(tf.float32, shape=[None, self.num_seqs, self.horizon],
                                       name='label')

    def after_build(self):
        with tf.name_scope('train'):
            self.loss = tf.reduce_mean(tf.square(self.y_pred - self.label_ph))
            self.optimizer = tf.train.AdamOptimizer(self.lr)

            self.train_op = self.optimizer.minimize(self.loss)

    def build(self):
        # build placeholders
        self.post_build()
        # build model
        with tf.variable_scope('GeoMAN'):
            with tf.variable_scope('encoder'):
                encoder_rnn = self.get_rnn_cell(self.hidden_size, 1 - self.dropout_rate)

                # shape -> [batch_size, num_seqs, x_dim, T]
                x_input = tf.transpose(self.x_ph, perm=[0, 1, 3, 2])
                # shape -> [batch_size, num_seqs, T]
                y_input = self.y_ph

                def encoder_loop_fn(prev_states, inputs):
                    """
                    :param prev_states: states of rnn
                    :param inputs: [batch_size, num_seqs, x_dim], [batch_size, num_seqs]
                    :return:
                    """
                    # shape -> [batch_size, num_seqs, x_dim], [batch_size, num_seqs]
                    x, y = inputs
                    # shape -> [batch_size, num_seqs, hidden_size * 2]
                    prev_state_concat = tf.concat(prev_states, axis=-1)

                    # shape -> [batch_size, num_seqs, x_dim]
                    alphas = self.local_attention(prev_state_concat, x_input, 'local_attention')
                    x_local = x * alphas

                    # shape -> [batch_size, num_seqs, num_seqs]
                    betas = self.global_attention(prev_state_concat, y_input, x_input, 'global_attention')
                    x_global = tf.expand_dims(y, axis=-2) * betas

                    # shape -> [batch_size, num_seqs, x_dim + num_seqs]
                    encoder_input = tf.concat([x_local, x_global], axis=-1)

                    en_h_stete, en_states = encoder_rnn.call(encoder_input, prev_states)

                    return en_states

                # shape -> [T, batch_size, num_seqs, x_dim]
                x_scan_var = tf.transpose(self.x_ph, perm=[2, 0, 1, 3])
                # shape -> [T, batch_size, num_seqs]
                y_scan_var = tf.transpose(self.y_ph, perm=[2, 0, 1])

                en_initial_states = [tf.zeros([tf.shape(self.x_ph)[0], self.num_seqs, self.hidden_size]),
                                     tf.zeros([tf.shape(self.x_ph)[0], self.num_seqs, self.hidden_size])]

                # list of rnn_states
                rnn_outputs = tf.scan(encoder_loop_fn,
                                      [x_scan_var, y_scan_var],
                                      en_initial_states)
                # shape -> [batch_size, num_seqs, T, hidden_size]
                encoder_h_states = tf.transpose(rnn_outputs[1], perm=[1, 2, 0, 3])

            with tf.variable_scope('inference'):
                infer_v = self.get_weights('infer_v', [self.hidden_size, 1])
                infer_w = self.get_weights('infer_w', [self.hidden_size * 2, self.hidden_size])
                infer_w_b = self.get_bias('infer_w_b', [self.hidden_size])
                infer_v_b = self.get_bias('infer_v_b', [1])

            with tf.variable_scope('decoder'):
                decoder_rnn = self.get_rnn_cell(self.hidden_size, 1 - self.dropout_rate)

                y_pred_ta = tf.TensorArray(tf.float32, size=self.T + self.horizon)
                # shape -> [batch_size, num_seqs, hidden_size]
                context_state = tf.reduce_mean(encoder_h_states, axis=-2, keepdims=False)
                de_states = en_initial_states
                y_pred = tf.zeros([tf.shape(self.x_ph)[0], self.num_seqs, 1])
                for t in range(self.T + self.horizon):
                    # shape -> [batch_size, num_seqs, hidden_size + 1]
                    de_input = tf.concat([y_pred, context_state], axis=-1)

                    de_h_state, de_states = decoder_rnn.call(de_input, de_states)

                    prev_state_concat = tf.concat(de_states, axis=-1)
                    # shape -> [batch_size, num_seqs, T, 1]
                    gammas = self.temporal_attention(prev_state_concat, encoder_h_states, 'temporal_attention')

                    # shape -> [batch_size, num_seqs, hidden_size]
                    context_state = tf.reduce_sum(gammas * encoder_h_states, axis=-2, keepdims=False)

                    y_pred = tensordot(tf.concat([context_state, de_h_state], axis=-1), infer_w) + infer_w_b
                    # shape -> [batch_size, num_seqs, 1]
                    y_pred = tensordot(tf.nn.tanh(y_pred), infer_v) + infer_v_b

                    y_pred_ta = y_pred_ta.write(t, y_pred)

                # shape -> [batch_size, num_seqs, T + horizon]
                y_preds = tf.squeeze(tf.transpose(y_pred_ta.stack(), perm=[1, 2, 0, 3]), axis=-1)

            self.y_pred = y_preds[:, :, -self.horizon:]
        # build training staff
        self.after_build()

    def local_attention(self, src_state, tgt_seqs, scope):
        """ Calculate local attention weights for current src_state with each sequence in tgt_seqs.
        :param src_states: [..., hidden_size * 2]
        :param tgt_seqs: [..., x_dim, T]
        :param scope:
        :return: [..., x_dim]
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # create variables
            att_v = self.get_weights('att_v', [self.T, 1])
            att_w = self.get_weights('att_w', [2 * self.hidden_size, self.T])
            att_u = self.get_weights('att_u', [self.T, self.T])
            att_b = self.get_bias('att_b', [self.T])

            # operations
            # shape -> [..., x_dim, T]
            e_logits = tf.expand_dims(tensordot(src_state, att_w), axis=-2) \
                       + tensordot(tgt_seqs, att_u) + att_b
            # shape -> [..., x_dim]
            e_logits = tf.squeeze(tensordot(tf.nn.tanh(e_logits), att_v) / tf.sqrt(float(self.T)), axis=-1)

            # shape -> [..., x_dim]
            weights = tf.nn.softmax(e_logits, axis=-1)

            return weights

    def global_attention(self, src_state, tgt_y_seqs, tgt_x_seqs, scope):
        """ Calculate global attention weights for current src_state with each sequence in tgt_y_seqs and tgt_x_seqs.
        :param src_state: [..., num_seqs, hidden_size * 2]
        :param tgt_y_seqs: [..., num_seqs, T]
        :param tgt_x_seqs: [..., num_seqs, x_dim, T]
        :param scope:
        :return: [..., num_seqs, num_seqs]
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # create variables
            att_v = self.get_weights('att_v', [self.T, 1])
            att_w = self.get_weights('att_w', [2 * self.hidden_size, self.T])
            att_y_u = self.get_weights('att_y_u', [self.T, self.T])
            att_x_w = self.get_weights('att_x_w', [self.x_dim, self.T])
            att_x_u = self.get_weights('att_x_u', [self.T, 1])
            att_b = self.get_bias('att_b', [self.T])

            # operations
            # shape -> [..., num_seqs, 1, T]
            h_part = tf.expand_dims(tensordot(src_state, att_w), axis=-2)

            # shape -> [..., num_seqs, x_dim]
            x_part = tf.squeeze(tensordot(tgt_x_seqs, att_x_u), axis=-1)
            # shape -> [..., 1, num_seqs, T]
            x_part = tf.expand_dims(tensordot(x_part, att_x_w), axis=-3)

            # shape -> [..., 1, num_seqs, T]
            y_part = tf.expand_dims(tensordot(tgt_y_seqs, att_y_u), axis=-3)

            # shape -> [..., num_seqs, num_seqs, T]
            e_logits = h_part + x_part + y_part + att_b
            # shape -> [..., num_seqs, num_seqs]
            e_logits = tf.squeeze(tensordot(e_logits, att_v), axis=-1)

            # shape -> [..., num_seqs, num_seqs]
            weights = tf.nn.softmax(e_logits, axis=-1)

            return weights

    def temporal_attention(self, src_state, tgt_states, scope):
        """ Calculate temporal attention weights for current src_state with each tgt_states.
        :param src_state: [..., hidden_size * 2]
        :param tgt_states: [..., T, hidden_size]
        :param scope:
        :return: attention_weights: [..., T, 1]
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # create variables
            att_v = self.get_weights('att_v', [self.hidden_size, 1])
            att_w = self.get_weights('att_w', [self.hidden_size * 2, self.hidden_size])
            att_u = self.get_weights('att_u', [self.hidden_size, self.hidden_size])
            att_b = self.get_bias('att_b', [self.hidden_size])

            # operations
            # shape -> [..., T, hidden_size]
            e_logits = tf.expand_dims(tensordot(src_state, att_w), axis=-2) \
                       + tensordot(tgt_states, att_u) \
                       + att_b
            # shape -> [..., T, 1]
            e_logits = tensordot(tf.nn.tanh(e_logits), att_v)

            # shape -> [..., T, 1]
            weights = tf.nn.softmax(e_logits, axis=-2)

            return weights
