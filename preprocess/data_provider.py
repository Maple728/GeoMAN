#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: GeoMAN
@time: 2019/11/5 21:01
@desc:
"""

import numpy as np


def window_rolling(origin_data, window_size):
    """
    :param origin_data: ndarray of [n_records, num_seqs, dim]
    :param window_size: window_size
    :return: [n_records - window_size + 1, window_size, num_seqs, dim]
    """
    n_records = len(origin_data)
    if n_records < window_size:
        return None

    data = origin_data[:, None]
    all_data = []
    for i in range(window_size):
        all_data.append(data[i: (n_records - window_size + i + 1)])

    # shape -> [n_records - window_size + 1, window_size, num_seqs, dim]
    rolling_data = np.hstack(all_data)

    return rolling_data


class DataProvider:
    def __init__(self, data_source, T, horizon, batch_size):
        self._data_source = data_source
        self._batch_size = batch_size
        self._T = T
        self._horizon = horizon

    def process_batch_data(self, record_datas):
        """

        :param record_datas: []
        :return:
        """

        pass


