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

    def _run_epoch(self, data_provider):
        pass

    def train_model(self, model, data_provider, max_epoch):
        with tf.Session() as sess:
            # build model
            model.build()

            # initialize variables
            sess.run([tf.global_variables_initializer()])
            for epoch_num in range(max_epoch):
                loss, preds, labels = self._run_epoch(data_provider)


