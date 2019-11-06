#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: GeoMAN
@time: 2019/11/6 16:42
@desc:
"""

import numpy as np
from preprocess.scaler import MinMaxScaler
from preprocess.data_source import StaticDataSource


class PEMS4DataLoader(object):
    name = 'PEMS04'
    data_filename = './datasets/PEMS04/pems04.npz'

    def __init__(self, config):
        self.config = config

    def read_raw_data(self):
        # [length, num_of_vertices (307), 3 (traffic flow, occupancy, speed)]
        records = np.load(self.data_filename)
        tgt_idx = 0
        x_dim = records.shape[-1]

        # split train, valid and test data set
        post_len = self.config['T']
        # for same with ASTGCN
        lens = len(records) - post_len
        train_split_idx = int(lens * 0.6) + post_len
        # 2993 points for test == 2993 + 11 (horizon 12)
        valid_split_idx = -2993 - 11
        train_records = records[:train_split_idx]
        valid_records = records[train_split_idx - post_len:valid_split_idx]
        test_records = records[valid_split_idx - post_len:]

        # scaling data
        # scaling features
        feat_scaler = MinMaxScaler()
        train_feats = feat_scaler.fit_scaling(train_records)
        valid_feats = feat_scaler.scaling(valid_records)
        test_feats = feat_scaler.scaling(test_records)

        # scaling target series
        tgt_scaler = MinMaxScaler()
        train_tgts = tgt_scaler.fit_scaling(train_records[:, :, tgt_idx])
        valid_tgts = feat_scaler.scaling(valid_records[:, :, tgt_idx])
        test_tgts = feat_scaler.scaling(test_records[:, :, tgt_idx])

        train_ds = StaticDataSource(x_dim, self.name, [train_feats, train_tgts])
        valid_ds = StaticDataSource(x_dim, self.name, [valid_feats, valid_tgts])
        test_ds = StaticDataSource(x_dim, self.name, [test_feats, test_tgts])
