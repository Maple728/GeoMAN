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
    def __init__(self, x_dim, ds_name, retrieve_data_callback, use_cache=True, cache_dir='cache'):
        self.x_dim = x_dim
        self.ds_name = ds_name
        self._use_cache = use_cache
        self._retrieve_data_callback = retrieve_data_callback
        self.is_cached = False

        # create cache dir
        if self._use_cache:
            self.cache_path = create_folder(cache_dir, self.ds_name)

    def load_partition_data(self):
        """Iterate data from callback function or disk cache.
        :return: [feat_arr, target_arr]
        """
        if self.is_cached:
            for i, record_data in enumerate(self._retrieve_data_callback()):
                if self._use_cache:
                    # cache data into disk
                    np.savez(os.path.join(self.cache_path, str(i) + '.npz'),
                             feat=record_data[0],
                             target=record_data[1])
                yield record_data
            if self._use_cache:
                self.is_cached = True
        else:
            for filename in os.listdir(self.cache_path):
                npzfile = np.load(os.path.join(self.cache_path, filename))
                yield [npzfile['feat'], npzfile['target']]
