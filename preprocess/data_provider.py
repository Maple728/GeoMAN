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
    def __init__(self, data_source, T, horizon, batch_size,
                 is_provide_label=True):
        self._data_source = data_source
        self._batch_size = batch_size
        self._T = T
        self._horizon = horizon
        self._is_provide_label = is_provide_label
        if is_provide_label:
            self._window_size = self._T + self._horizon
        else:
            self._window_size = self._T

    def process_model_input(self, item_data):
        """

        :param item_data: [n_items, num_seqs, T + horizon, dim]
        :return:
        """

        x = item_data[:, :, : self._T, :]
        y = item_data[:, :, : self._T, self._data_source.tgt_idx]
        if self._is_provide_label is True:
            label_y = item_data[:, :, self._T:, self._data_source.tgt_idx]
            return x, y, label_y
        else:
            return x, y

    def iterate_batch_data_with_label(self):
        # record_data shape -> [n_records, num_seqs, dim]
        for record_data in self._data_source.load_partition_data():
            # process
            # shape -> [n_items, num_seqs, window_size, dim]
            item_data = np.transpose(window_rolling(record_data, self._window_size),
                                     [0, 2, 1, 3])
            inputs = self.process_model_input(item_data)

            yield split2batch_data(inputs, self._batch_size, keep_remainder=True)
