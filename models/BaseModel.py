#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: GeoMAN
@time: 2019/11/5 15:50
@desc:
"""


class BaseModel(object):

    def train(self, sess, batch_data, lr):
        loss, pred, real = None, None, None

        return loss, pred, real

    def predict(self, sess, batch_data, lr = None):
        loss, pred, real = None, None, None
        return loss, pred, real
