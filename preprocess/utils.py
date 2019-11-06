#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: GeoMAN
@time: 2019/11/6 15:43
@desc:
"""
import numpy as np


def window_rolling(origin_data, window_size):
    """
    :param origin_data: ndarray of [n_records, ...]
    :param window_size: window_size
    :return: [n_records - window_size + 1, window_size, ...]
    """
    n_records = len(origin_data)
    if n_records < window_size:
        return None

    data = origin_data[:, None]
    all_data = []
    for i in range(window_size):
        all_data.append(data[i: (n_records - window_size + i + 1)])

    # shape -> [n_records - window_size + 1, window_size, ...]
    rolling_data = np.hstack(all_data)

    return rolling_data


def split2batch_data(arrs, batch_size, keep_remainder=True):
    """

    :param arrs: a list of [n_items, ...]
    :param batch_size:
    :param keep_remainder: Discard the remainder if False, otherwise keep it.
    :return:
    """
    if arrs is None or len(arrs) == 0:
        return

    idx = 0
    n_items = len(arrs[0])
    while idx < n_items:
        if idx + batch_size > n_items and keep_remainder is False:
            return
        next_idx = min(idx + batch_size, n_items)
        yield [arr[idx: next_idx] for arr in arrs]

        # update idx
        idx = next_idx
