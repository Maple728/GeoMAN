#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: GeoMAN
@time: 2019/11/7 21:09
@desc:
"""

from models.GeoMAN import GeoMAN
from trainer import Trainer
from preprocess.data_loader import PEMS4DataLoader
from preprocess.data_provider import DataProvider


if __name__ == '__main__':
    max_epoch = 300
    config = dict()
    config['num_seqs'] = 307
    config['T'] = 24
    config['horizon'] = 12
    config['x_dim'] = 3
    config['hidden_size'] = 64
    config['batch_size'] = 16

    # load data
    data_loader = PEMS4DataLoader(config)
    train_ds, valid_ds, test_ds = data_loader.read_raw_data()

    train_data_provider = DataProvider(train_ds, config['T'], config['horizon'], config['batch_size'])
    valid_data_provider = DataProvider(valid_ds, config['T'], config['horizon'], config['batch_size'])
    test_data_provider = DataProvider(test_ds, config['T'], config['horizon'], config['batch_size'])

    model = GeoMAN(config)
    trainer = Trainer()
    trainer.train_model(model, max_epoch,
                        train_data_provider, valid_data_provider, test_data_provider)
