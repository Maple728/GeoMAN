#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: GeoMAN
@time: 2019/11/6 14:58
@desc:
"""

import abc


class DataSource(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, x_dim, ds_name):
        self.x_dim = x_dim
        self.ds_name = ds_name

    @abc.abstractmethod
    def load_partition_data(self):
        pass


class StaticDataSource(DataSource):
    name = 'PeMSD4'

    def __init__(self, x_dim, ds_name, record_data):
        super(StaticDataSource, self).__init__(x_dim, ds_name)
        self._record_data = record_data

    def load_partition_data(self):
        return self._record_data
