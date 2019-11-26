#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: GeoMAN
@time: 2019/11/5 15:48
@desc:
"""

import os
from functools import reduce
from operator import mul

import numpy as np
import tensorflow as tf


def get_num_params():
    num_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    return num_params


class Trainer(object):
    def __init__(self, base_dir='logs'):
        self._tfb_dir = os.path.join(base_dir, 'tfb')
        self._model_dir = os.path.join(base_dir, 'checkpoints')

    def make_config_string(self, config):
        str_config = ''
        for k, v in config.items():
            str_config += k[:3] + str(v) + '-'
        return str_config[:-1]

    def _run_epoch(self, sess, model, data_provider, lr, is_train):
        if is_train:
            run_func = model.train
        else:
            run_func = model.predict
        loss_list = []
        pred_list = []
        real_list = []
        for batch_data in data_provider.iterate_batch_data(True):
            loss, pred, real = run_func(sess, batch_data, lr)

            loss_list.append(loss)
            pred_list.append(pred)
            real_list.append(real)

        # shape -> [n_items, num_seqs, horizon]
        epoch_preds = np.concatenate(pred_list, axis=0)
        epoch_reals = np.concatenate(real_list, axis=0)

        epoch_avg_loss = np.mean(loss_list)
        # inverse scaling data
        epoch_preds = data_provider.data_source.scaler.inverse_scaling(epoch_preds)
        epoch_reals = data_provider.data_source.scaler.inverse_scaling(epoch_reals)

        return epoch_avg_loss, epoch_preds, epoch_reals

    def train_model(self, model, max_epoch,
                    train_data_provider, valid_data_provider, test_data_provider):
        lr = 0.001

        with tf.Session() as sess:
            # build model
            model.build()

            print('----------Trainable parameter count:', get_num_params())
            # initialize variables
            sess.run([tf.global_variables_initializer()])

            best_valid_loss = float('inf')
            for epoch_num in range(max_epoch):
                # train
                self._run_epoch(sess, model, train_data_provider,
                                lr, is_train=True)

                # valid
                loss, _, _ = self._run_epoch(sess, model, valid_data_provider,
                                             lr, is_train=False)
                if loss < best_valid_loss:
                    best_valid_loss = loss

                    # TODO save model

                    # test
                    loss, preds, labels = self._run_epoch(sess, model, test_data_provider,
                                                          lr, is_train=False)
                    metrics = test_data_provider.data_source.get_metrics(preds, labels)
                    # str_metrics = reduce(lambda acc, m: acc + ', ' + str(m[0]) + ': ' + str(m[1]), metrics, '')
                    str_metrics = str(metrics)
                    print('Epoch', epoch_num, 'Test Loss:', loss, str_metrics)

