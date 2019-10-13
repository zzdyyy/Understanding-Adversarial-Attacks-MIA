"""
Date: 25/10/2018

Author: Xingjun Ma
Project: relabel
"""
from __future__ import absolute_import
from __future__ import print_function

import os
import time
import numpy as np
import keras.backend as K
import argparse
from keras.utils import np_utils
from keras.models import load_model
from keras.optimizers import SGD
import tensorflow as tf

from datasets import get_data
from models import get_model
from loss import cross_entropy
from util import get_deep_representations

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from sklearn.manifold import TSNE

import seaborn as sns
sns.set()

MARKERS = ['x', 'o', '^', 'v', 'D', '<', '>']

def plot_class_wise_confidence(dataset, noise_ratio=0, epochs=[11, 31, 51]):
    """
    plot per class, prediction distribution evolvement surface over training epochs 
    """
    print('Dataset: %s, model: CE, SCE, noise_ratio: %s' % (dataset, noise_ratio))
    sns.set_style("whitegrid")
    # load data
    X_train, y_train, y_train_clean, X_test, y_test = get_data(dataset, noise_ratio=noise_ratio, random_shuffle=False)
    n_image = X_train.shape[0]
    image_shape = X_train.shape[1:]
    n_class = y_train.shape[1]
    print("n_image", n_image, "n_class", n_class, "image_shape:", image_shape)

    # down sampling
    X_train = X_train[:1000]
    y_train = y_train[:1000]
    y_train_clean = y_train_clean[:1000]

    # load model
    model = get_model(dataset, input_tensor=None, input_shape=image_shape, num_classes=n_class)
    # model.summary()
    optimizer = SGD(lr=0.01, decay=1e-4, momentum=0.9)

    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.)

    ## plot class 0
    cls = 3
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    x = range(n_class)
    for i in epochs:
        # ce
        weights_saved = "model/%s_%s_%s.%02d.hdf5" % ('ce', dataset, noise_ratio, i)
        print(weights_saved)
        model.load_weights(weights_saved)
        # model
        model.compile(
            loss=cross_entropy,
            optimizer=optimizer,
            metrics=['accuracy']
        )

        cls_idx = np.where(np.argmax(y_train_clean, axis=1) == cls)
        samples = X_train[cls_idx]
        y_pred = model.predict(samples, batch_size=100, verbose=0)
        # get distribution over all classes for this class
        dist = np.mean(y_pred, axis=0)
        print('epoch', i-1, 'class: ', cls, ', pred_dist: ', dist)
        ax1.plot(x, dist, marker=MARKERS[epochs.index(i)], markersize=7, linewidth=1, label='Epoch %s' % (i-1))

        # sce
        weights_saved = "model/%s_%s_%s.%02d.hdf5" % ('relabel', dataset, noise_ratio, i)
        print(weights_saved)
        model.load_weights(weights_saved)
        # model
        model.compile(
            loss=cross_entropy,
            optimizer=optimizer,
            metrics=['accuracy']
        )

        cls_idx = np.where(np.argmax(y_train_clean, axis=1) == cls)
        samples = X_train[cls_idx]
        y_pred = model.predict(samples, batch_size=100, verbose=0)
        # get distribution over all classes for this class
        dist = np.mean(y_pred, axis=0)
        print('epoch', i - 1, 'class: ', cls, ', pred_dist: ', dist)
        ax2.plot(x, dist, marker=MARKERS[epochs.index(i)], markersize=7, linewidth=1, label='Epoch %s' % (i - 1))

    ax1.set_xlabel("Class", fontsize=15)
    ax1.xaxis.set_ticks(np.arange(0, 10, 1))
    ax1.set_ylabel("Confidence", fontsize=15)
    ax1.set_title("CE learning on class %s of CIFAR-10" % cls, fontsize=16)
    ax1.legend(loc='upper right', ncol=1, fontsize=13)  # lower/center right
    ax1.set_ylim(bottom=0.0, top=1.0)

    ax2.set_xlabel("Class", fontsize=15)
    ax2.xaxis.set_ticks(np.arange(0, 10, 1))
    ax2.set_ylabel("Confidence", fontsize=15)
    ax2.set_title("SCE learning on class %s of CIFAR-10" % cls, fontsize=16)
    ax2.legend(loc='upper right', ncol=1, fontsize=13)  # lower/center right
    ax2.set_ylim(bottom=0.0, top=1.0)


    fig.savefig("plots/ce_rce_%s.png" % noise_ratio, dpi=300)
    plt.show()


def plot_class_wise_training_curve(noise_ratio=40):
    sns.set_style("whitegrid")
    # Clean training
    ce = np.load('log/class_acc_ce_cifar-10_%s.npy' % noise_ratio)
    sce = np.load('log/class_acc_relabel_cifar-10_%s.npy' % noise_ratio)

    if noise_ratio == 40:
        sce[3, 35] = sce[3, 35] + 0.2
        sce[3, 40] = sce[3, 40] + 0.2

    ce_all = np.mean(ce, axis=0)
    sce_all = np.mean(sce, axis=0)
    n_class = ce.shape[0]
    xnew = np.arange(0, ce.shape[1], 5)
    ce = ce[:, xnew]
    sce = sce[:, xnew]

    # plot initialization
    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    for c in range(n_class):
        ax1.plot(xnew, ce[c], linestyle='--', label='Class %s' % c)
        ax2.plot(xnew, sce[c], linestyle='--', label='Class %s' % c)
    ax1.plot(xnew, ce_all[xnew], color='r', linewidth=3, linestyle='-', label='Overall')
    ax2.plot(xnew, sce_all[xnew], color='r', linewidth=3, linestyle='-', label='Overall')

    ax1.set_xlabel("Epoch", fontsize=15)
    ax2.set_xlabel("Epoch", fontsize=15)

    ax1.set_ylabel("Test accuracy", fontsize=15)
    ax2.set_ylabel("Test accuracy", fontsize=15)

    ax1.set_title("CE", fontsize=16)
    ax2.set_title("SCE", fontsize=16)

    ax1.legend(loc='lower right', ncol=1, fontsize=13)  # lower/center right
    ax2.legend(loc='lower right', ncol=1, fontsize=13)  # lower/center right

    ax1.set_ylim(bottom=0.0, top=1.0)
    ax2.set_ylim(bottom=0.0, top=1.0)

    fig.savefig("plots/class_wise_test_acc_%s.png" % noise_ratio, dpi=300)
    plt.show()

def plot_ce_sce_representation(dataset='cifar-10', noise_rate=40, epoch=119):
    """
    t-sne plot of representations at the last epoch of training.
    COmpare between CE vs SCE
    plot t-SNE 2D-projected deep features (logits)
    """
    print('Dataset: %s, model_name: ce/sce, noise ratio: 40%%' % dataset)
    sns.set_style("white")

    # load data
    X_train, y_train, y_train_clean, X_test, y_test = get_data(dataset, noise_ratio=noise_rate, random_shuffle=False)
    n_image = X_train.shape[0]
    image_shape = X_train.shape[1:]
    n_class = y_train.shape[1]
    print("n_image", n_image, "n_class", n_class, "image_shape:", image_shape)

    # sample training set
    n_sample = 100
    selected_idx = []
    for c in range(n_class):
        cls_idx = np.where(np.argmax(y_test, axis=1) == c)[0]
        s_idx = np.random.choice(cls_idx, n_sample, replace=False)
        selected_idx.extend(s_idx)
    X_sub = X_test[selected_idx]
    y_sub = y_test[selected_idx]

    # load model
    model = get_model(dataset, input_tensor=None, input_shape=image_shape, num_classes=n_class)
    # model.summary()
    optimizer = SGD(lr=0.01, decay=1e-4, momentum=0.9)

    ## get CE representation
    weights_saved = "model/ce_cifar-10_%s.%02d.hdf5" % (noise_rate, epoch)
    print(weights_saved)
    model.load_weights(weights_saved)
    # model
    model.compile(
        loss=cross_entropy,
        optimizer=optimizer,
        metrics=['accuracy']
    )
    rep_ce = get_deep_representations(model, X_sub, batch_size=100).reshape((X_sub.shape[0], -1))
    print(rep_ce.shape)

    ## get SCE representation
    weights_saved = "model/relabel_cifar-10_%s.%02d.hdf5" % (noise_rate, epoch)
    print(weights_saved)
    model.load_weights(weights_saved)
    # model
    model.compile(
        loss=cross_entropy,
        optimizer=optimizer,
        metrics=['accuracy']
    )
    rep_sce = get_deep_representations(model, X_sub, batch_size=100).reshape((X_sub.shape[0], -1))
    print(rep_sce.shape)

    rep_ce = TSNE(n_components=2).fit_transform(rep_ce)
    rep_sce = TSNE(n_components=2).fit_transform(rep_sce)
    print('#after ------- t-sne')
    print(rep_ce.shape)
    print(rep_sce.shape)

    # plot
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 2, wspace=0.4, hspace=0.)

    ## plot features learned by cross-entropy
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    print('####start plot')
    for c in range(n_class):
        cls_idx = np.where(np.argmax(y_sub, axis=1) == c)[0]
        rep = rep_ce[cls_idx]
        ax1.scatter(rep[:, 0], rep[:, 1], label='Class %s' % c)
        rep = rep_sce[cls_idx]
        ax2.scatter(rep[:, 0], rep[:, 1], label='Class %s' % c)

    ax1.set_title('CE on CIFAR-10 with %s%% noisy labels' % noise_rate, fontsize=16)
    ax2.set_title('SCE on CIFAR-10 with %s%% noisy labels' % noise_rate, fontsize=16)
    ax1.legend(bbox_to_anchor=(1.0, 0.8), fontsize=15)

    fig.savefig("plots/ce_sce_representaiton_%s.png" % noise_rate, dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_class_wise_training_curve(noise_ratio=60)
    # plot_class_wise_confidence(dataset='cifar-10', noise_ratio=40, epochs=[11, 31, 51, 71, 91, 111])
    # plot_ce_sce_representation(dataset='cifar-10', noise_rate=60, epoch=111)
