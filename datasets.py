"""
Date: 9/08/2018

Author: Xingjun Ma
Project: elastic_adv_defense
"""
from __future__ import absolute_import
from __future__ import print_function
import warnings
import os, time
import numpy as np
import scipy.io as sio
from subprocess import call
from keras.datasets import mnist, cifar10
from keras.utils import np_utils
import keras
import glob
import cv2
import imageio
from global_config import *


STDEVS = {
    'mnist': {'fgsm': 0.3, 'bim-a': 0.111, 'pgd': 0.27, 'cw-l2': 0.207},
    'cifar-10': {'fgsm': 0.031, 'bim-a': 0.031, 'pgd': 0.023, 'cw-l2': 0.023},
    'svhn': {'fgsm': 0.133, 'bim-a': 0.0155, 'pgd': 0.095, 'cw-l2': 0.008},
    'dr': {'fgsm': 0.0157, 'bim-a': 0.00176, 'bim-b': np.nan, 'cw-l2': np.nan},
    'cxr': {'fgsm': 0.0235, 'bim-a': 0.00314, 'bim-b': np.nan, 'cw-l2': np.nan},
    'derm': {'fgsm': 0.0149, 'bim-a': 0.00242, 'bim-b': np.nan, 'cw-l2': np.nan},
}


def get_data(dataset='mnist', clip_min=-0.5, clip_max=0.5, onehot=True, path='data/', split_traintest=True, load_feat=None):
    """
    images in [-0.5, 0.5] (instead of [0, 1]) which suits C&W attack 
    images in [-1, 1] with Finlayson's model
    :param dataset:
    :param split_traintest: spilt train/val for
    :param load_feat: if provided with attack name, load the hidden layer feature for that adv data; or load raw images
    :return: 
    """
    if not os.path.exists(path):
        os.makedirs(path)

    if dataset == 'imagenet':
        flist = sorted(glob.glob('data/imagenet/*.jpg'))
        X = map(lambda path: cv2.resize(imageio.imread(path), (224, 224)), flist)
        X = list(X)
        X = np.stack(X)
        X = keras.applications.resnet50.preprocess_input(X)
        Y = keras.utils.to_categorical([281, 281, 281, 250, 250, 281, 281, 250, 281, 281, 250, 281, 250], 1000)
        return X[:0], Y[:0], X, Y

    if dataset in ['dr', 'cxr', 'derm', 'cxr056', 'cxr0456', 'cxr05']:
        if load_feat is not None:
            X_all = np.load('data/' + ADV_PREFIX + 'feat_%s_%s.npy' % (dataset, load_feat))
        else:
            X_all = np.load('adversarial_medicine/numpy_to_share/%s/val_test_x.npy' % dataset).astype('float32')
            if X_all.shape[-1] == 1:
                X_all = np.repeat(X_all, 3, axis=-1)
            keras.applications.inception_resnet_v2.preprocess_input(X_all)  # transform value range to [-1, 1]
        Y_all = np.load('adversarial_medicine/numpy_to_share/%s/val_test_y.npy' % dataset)
        if not onehot:
            Y_all = np.argmax(Y_all, axis=1)

        if split_traintest:
            correct_idx, train_idx, test_idx = np.load('data/' + ADV_PREFIX + 'split_%s.npy' % dataset, allow_pickle=True)
            X_train, Y_train = X_all[train_idx], Y_all[train_idx]
            X_test, Y_test = X_all[test_idx], Y_all[test_idx]
        else:
            X_train, Y_train = X_all[:0, ...], Y_all[:0, ...]
            X_test, Y_test = X_all, Y_all

        print("X_train:", X_train.shape)
        print("Y_train:", Y_train.shape)
        print("X_test:", X_test.shape)
        print("Y_test", Y_test.shape)

        return X_train, Y_train, X_test, Y_test

    if dataset == 'mnist':
        # the data, shuffled and split between train and test sets
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        # reshape to (n_samples, 28, 28, 1)
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)
    elif dataset == 'cifar-10':
        # the data, shuffled and split between train and test sets
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    elif dataset == 'svhn':
        if not os.path.isfile(os.path.join(path, "svhn_train.mat")):
            print('Downloading SVHN train set...')
            call(
                "curl -o data/svhn_train.mat "
                "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                shell=True
            )
        if not os.path.isfile(os.path.join(path, "svhn_test.mat")):
            print('Downloading SVHN test set...')
            call(
                "curl -o data/svhn_test.mat "
                "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                shell=True
            )
        train = sio.loadmat(os.path.join(path,'svhn_train.mat'))
        test = sio.loadmat(os.path.join(path, 'svhn_test.mat'))
        X_train = np.transpose(train['X'], axes=[3, 0, 1, 2])
        X_test = np.transpose(test['X'], axes=[3, 0, 1, 2])
        # reshape (n_samples, 1) to (n_samples,) and change 1-index
        # to 0-index
        y_train = np.reshape(train['y'], (-1,)) - 1
        y_test = np.reshape(test['y'], (-1,)) - 1
    else:
        print("Add new type of dataset here such as cifar-100.")
        return

    # cast pixels to floats, normalize to [0, 1] range
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train = (X_train / 255.0) - (1.0 - clip_max)
    X_test = (X_test / 255.0) - (1.0 - clip_max)

    n_class = np.max(y_train) + 1

    # one-hot-encode the labels
    if onehot:
        Y_train = np_utils.to_categorical(y_train, n_class)
        Y_test = np_utils.to_categorical(y_test, n_class)
    else:
        Y_train = y_train
        Y_test = y_test

    print("X_train:", X_train.shape)
    print("Y_train:", Y_train.shape)
    print("X_test:", X_test.shape)
    print("Y_test", Y_test.shape)

    return X_train, Y_train, X_test, Y_test


def get_noisy_samples(X_test, X_test_adv, dataset, attack, clip_min=-0.5, clip_max=0.5):
    """
    TODO
    :param X_test:
    :param X_test_adv:
    :param dataset:
    :param attack:
    :return:
    """
    if attack in ['jsma', 'cw-l0']:
        X_test_noisy = np.zeros_like(X_test)
        for i in range(len(X_test)):
            # Count the number of pixels that are different
            nb_diff = len(np.where(X_test[i] != X_test_adv[i])[0])
            # Randomly flip an equal number of pixels (flip means move to max
            # value of 1)
            X_test_noisy[i] = flip(X_test[i], nb_diff, clip_max)
    else:
        warnings.warn("Important: using pre-set Gaussian scale sizes to craft noisy "
                      "samples. You will definitely need to manually tune the scale "
                      "according to the L2 print below, otherwise the result "
                      "will inaccurate. In future scale sizes will be inferred "
                      "automatically. For now, manually tune the scales around "
                      "mnist: L2/20.0, cifar: L2/54.0, svhn: L2/60.0")
        # Add Gaussian noise to the samples
        # print(STDEVS[dataset][attack])
        X_test_noisy = np.minimum(
            np.maximum(
                X_test + np.random.normal(loc=0, scale=STDEVS[dataset][attack],
                                          size=X_test.shape),
                clip_min
            ),
            clip_max
        )

    return X_test_noisy

def flip(x, nb_diff, clip_max=0.5):
    """
    Helper function for get_noisy_samples
    :param x:
    :param nb_diff:
    :return:
    """
    original_shape = x.shape
    x = np.copy(np.reshape(x, (-1,)))
    candidate_inds = np.where(x < clip_max)[0]
    assert candidate_inds.shape[0] >= nb_diff
    inds = np.random.choice(candidate_inds, nb_diff)
    x[inds] = clip_max

    return np.reshape(x, original_shape)
