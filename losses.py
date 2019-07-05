# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

from keras import backend as K
import tensorflow as tf


def cross_entropy(y_true, y_pred):
    y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)
    y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
    return -tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)

def cross_entropy_with_logits(temperature=1.):
    def loss(y_true, y_pred):
        return tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred / temperature)
    return loss
