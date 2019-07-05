# -*- coding: utf-8 -*-
import os
import numpy as np
import keras.backend as K
from keras.callbacks import Callback, LearningRateScheduler
import tensorflow as tf


class LoggerCallback(Callback):
    """
    Log train/val loss and acc into file for later plots.
    """
    def __init__(self, sess, model, X_test, Y_test, dataset, epochs):
        super(LoggerCallback, self).__init__()
        self.sess = sess
        self.model = model
        self.X_test = X_test
        self.Y_test = Y_test
        self.dataset = dataset
        self.epochs = epochs

        self.train_loss = []
        self.test_loss = []
        self.train_acc = []
        self.test_acc = []

        self.log_path = './log'
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def on_epoch_end(self, epoch, logs={}):
        tr_acc = logs.get('acc')
        tr_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        val_acc = logs.get('val_acc')

        self.train_loss.append(tr_loss)
        self.test_loss.append(val_loss)
        self.train_acc.append(tr_acc)
        self.test_acc.append(val_acc)

        file_name = 'log/shuffle_loss_acc_%s.npy' % self.dataset
        np.save(file_name, np.stack((np.array(self.train_loss), np.array(self.test_loss),
                                     np.array(self.train_acc), np.array(self.test_acc))))

        return


def get_lr_scheduler(dataset):
    """
    customerized learning rate decay for training with clean labels.
     For efficientcy purpose we use large lr for noisy data.
    :param dataset: 
    :param noise_ratio:
    :return: 
    """
    if dataset in ['mnist', 'svhn']:
        def scheduler(epoch):
            if epoch > 40:
                return 0.01
            else:
                return 0.1
        return LearningRateScheduler(scheduler)
    elif dataset in ['cifar-10']:
        def scheduler(epoch):
            if epoch > 100:
                return 0.0001
            elif epoch > 60:
                return 0.001
            else:
                return 0.01
        return LearningRateScheduler(scheduler)
    elif dataset in ['cifar-100']:
        def scheduler(epoch):
            if epoch > 160:
                return 0.0001
            elif epoch > 80:
                return 0.001
            else:
                return 0.01
        return LearningRateScheduler(scheduler)
