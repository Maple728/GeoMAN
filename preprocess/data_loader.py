#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: GeoMAN
@time: 2019/11/6 16:42
@desc:
"""

import numpy as np
from preprocess.scaler import MinMaxScaler, MinMaxSingletonScaler
from preprocess.data_source import DataSource
from preprocess.utils import normal_rmse_np, normal_mae_np


def metrics(preds, labels):
    mae = normal_mae_np(preds, labels)
    rmse = normal_rmse_np(preds, labels)
    return {'MAE': mae, 'RMSE': rmse}


class PEMS4DataLoader(object):
    name = 'PEMS04'
    data_filename = './datasets/PEMS04/pems04.npz'

    def __init__(self, config):
        self.config = config

    def read_raw_data(self):
        tgt_idx = 0
        # [length, num_of_vertices (307), 3 (traffic flow, occupancy, speed)]
        records = np.load(self.data_filename)['data']

        x_dim = records.shape[2]
        num_seqs = records.shape[1]

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
        tgt_scaler = MinMaxSingletonScaler()
        train_tgts = tgt_scaler.fit_scaling(train_records[:, :, tgt_idx])
        valid_tgts = tgt_scaler.scaling(valid_records[:, :, tgt_idx])
        test_tgts = tgt_scaler.scaling(test_records[:, :, tgt_idx])

        def get_retrieve_data_callback(data):
            def func():
                yield data
            return func

        train_ds = DataSource(x_dim, num_seqs, self.name + '_train',
                              metric_callback=metrics,
                              retrieve_data_callback=get_retrieve_data_callback([train_feats, train_tgts]),
                              scaler=tgt_scaler, use_cache=True)
        valid_ds = DataSource(x_dim, num_seqs, self.name + '_valid',
                              metric_callback=metrics,
                              retrieve_data_callback=get_retrieve_data_callback([valid_feats, valid_tgts]),
                              scaler=tgt_scaler, use_cache=True)
        test_ds = DataSource(x_dim, num_seqs, self.name + '_test',
                             metric_callback=metrics,
                             retrieve_data_callback=get_retrieve_data_callback([test_feats, test_tgts]),
                             scaler=tgt_scaler, use_cache=True)

        return train_ds, valid_ds, test_ds
