#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: GeoMAN
@time: 2019/11/6 14:58
@desc:
"""
import os
import numpy as np

from preprocess.utils import create_folder


class DataSource(object):
    def __init__(self, x_dim, num_seqs, ds_name,
                 metric_callback, retrieve_data_callback,
                 scaler=None, use_cache=True, cache_dir='cache'):
        # assign parameters
        self._x_dim = x_dim
        self._num_seqs = num_seqs
        self._ds_name = ds_name
        self._use_cache = use_cache

        self._metric_callback = metric_callback
        self._retrieve_data_callback = retrieve_data_callback

        self._scaler = scaler

        self.is_cached = False

        # create cache dir
        if self._use_cache:
            self.cache_path = create_folder(cache_dir, self._ds_name)

    @property
    def scaler(self):
        return self._scaler

    @property
    def x_dim(self):
        return self._x_dim

    @property
    def num_seqs(self):
        return self._num_seqs

    def get_metrics(self, preds, labels):
        return self._metric_callback(preds, labels)

    def load_partition_data(self):
        """Iterate data from callback function or disk cache.
        :return: [feat_arr, target_arr]
        """
        if self.is_cached:
            for filename in os.listdir(self.cache_path):
                npzfile = np.load(os.path.join(self.cache_path, filename))
                yield (npzfile['feat'], npzfile['target'])
        else:
            for i, record_data in enumerate(self._retrieve_data_callback()):
                if self._use_cache:
                    # cache data into disk
                    np.savez(os.path.join(self.cache_path, str(i) + '.npz'),
                             feat=record_data[0],
                             target=record_data[1])
                yield record_data
            if self._use_cache:
                self.is_cached = True
