#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: GeoMAN
@time: 2019/11/5 21:01
@desc:
"""

import numpy as np

from preprocess.utils import window_rolling, split2batch_data


class DataProvider:
    def __init__(self, data_source, T, horizon, batch_size):
        self._data_source = data_source
        self._batch_size = batch_size
        self._T = T
        self._horizon = horizon

    def _process_model_input(self, feat_data, target_data, provide_label):
        """

        :param feat_data: [n_items, num_seqs, window_size, dim]
        :param target_data: [n_items, num_seqs, window_size]
        :param provide_label:
        :return:
        """

        x = feat_data[:, :, : self._T]
        y = target_data[:, :, : self._T]
        if provide_label:
            label_y = target_data[:, :, self._T:]
            return x, y, label_y
        else:
            return x, y

    def iterate_batch_data(self, provide_label=True):
        """Get batch model input of one epoch.
        :param provide_label: return values with label if True
        :return:
        """
        if provide_label:
            window_size = self._T + self._horizon
        else:
            window_size = self._T
        # record_data shape -> [n_records, num_seqs, dim]
        for feat_data, target_data in self._data_source.load_partition_data():
            # process
            # shape -> [n_items, num_seqs, window_size, dim]
            feat_item_data = np.transpose(window_rolling(feat_data, window_size), [0, 2, 1, 3])
            # shape -> [n_items, num_seqs, window_size]
            target_item_data = np.transpose(window_rolling(target_data, window_size), [0, 2, 1, 3])
            inputs = self._process_model_input(feat_item_data, target_item_data, provide_label)

            yield split2batch_data(inputs, self._batch_size, keep_remainder=True)
