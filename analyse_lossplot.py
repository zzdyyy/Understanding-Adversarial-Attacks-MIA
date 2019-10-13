"""Detect adv/clean from the hidden feature"""
from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
from datasets import get_data
from models import get_model
import numpy as np
import sklearn.metrics
import keras.backend as K
import keras
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from global_config import *
import glob
from tqdm import tqdm

import imageio
import cv2

from vis.visualization import visualize_saliency
from vis.utils import utils
from cleverhans.evaluation import batch_eval

DATASETS = ['dr', 'cxr', 'derm']
ATTACKS = ['fgsm', 'bim', 'jsma', 'cw-l2', 'clean']
TEST_SIZE = {'dr': 0.7, 'cxr': 0.7, 'derm': 0.5}

CLIP_MIN = {'mnist': -0.5, 'cifar': -0.5, 'svhn': -0.5, 'dr': -1.0, 'cxr': -1.0, 'derm': -1.0, 'imagenet':-128}
CLIP_MAX = {'mnist':  0.5, 'cifar':  0.5, 'svhn':  0.5, 'dr':  1.0, 'cxr':  1.0, 'derm':  1.0, 'imagenet':128}

def solve_name_controdiction(model):
    for name in map(lambda x: x.__class__.__name__, model.layers):
        K.get_uid(name)
    K.get_uid('input')
    K.get_uid('input')


def analyze(args):
    assert args.dataset in ['mnist', 'cifar-10', 'svhn', 'dr', 'cxr', 'derm', 'imagenet'], \
        "Dataset parameter must be either 'mnist', 'cifar-10', 'svhn', 'dr', 'cxr', or 'derm'"

    # load feature/label data

    if args.dataset == 'imagenet':
        flist = glob.glob('data/imagenet/*.jpg')
        X = map(lambda path: cv2.resize(imageio.imread(path), (224, 224)), flist)
        X = list(X)
        X = np.stack(X)
        X = keras.applications.resnet50.preprocess_input(X)
        model = keras.applications.resnet50.ResNet50(include_top=True)
        layer_idx = utils.find_layer_idx(model, 'fc1000')
        y = model.predict(X).argmax(-1) #[248, 281, 281, 248, 281]
    elif args.dataset == 'mnist':
        _, _, X, y = get_data(args.dataset, onehot=False, split_traintest=False)  # clean image
        model = get_model(args.dataset, softmax=True)
        layer_idx = utils.find_layer_idx(model, 'dense_2')
        solve_name_controdiction(model)
    else:
        _, _, X, y = get_data(args.dataset, onehot=False, split_traintest=False)  # clean image

        model = get_model(args.dataset, softmax=True)
        layer_idx = utils.find_layer_idx(model, 'dense_2')  # 'dense_nosoftmax')
        solve_name_controdiction(model)

    x_in = model.input
    y_pred = model.output
    y_true = keras.backend.placeholder(shape=y_pred.shape, dtype='float32')
    loss = keras.losses.categorical_crossentropy(y_true, y_pred)
    x_grad, = keras.backend.gradients(loss, x_in)
    get_grad = keras.backend.function([x_in, y_true], [x_grad])
    get_loss = keras.backend.function([x_in, y_true], [loss])

    def reg(x):
        if x.shape[-1] == 1:
            x = np.tile(x, [1, 1, 3])
        return (x - x.min()) / (x.max() - x.min())


    plot_range = {
        'imagenet': slice(0, None),  # 2
        'derm': np.concatenate([ np.where(y>0)[0][-5:], np.where(y<1)[0][:5]  ]),   # -1
        'dr': np.concatenate([ np.where(y>0)[0][-5:], np.where(y<1)[0][:5]  ]),   # -6
        'cxr': np.concatenate([ np.where(y>0)[0][-5:], np.where(y<1)[0][:5]  ]),   #-
        'mnist': np.where(y == 6)[0][:10],
    }


    for i, (img, label) in enumerate(zip(X[plot_range[args.dataset]], y[plot_range[args.dataset]])):
        grads, = get_grad([img[None, ...], keras.utils.to_categorical(label, int(y_true.shape[1]))])
        grads = np.abs(grads)
        sort = np.argsort(grads.reshape([-1]))
        ind = np.unravel_index(np.argsort(grads, axis=None)[-2:], grads.shape)
        _, (h1, h2), (w1, w2), (c1, c2) = ind

        delta = 0.3 / 2 * (CLIP_MAX[args.dataset] - CLIP_MIN[args.dataset])
        # assert img[h1, w1, c1] - delta >= CLIP_MIN[args.dataset] and img[h1, w1, c1] + delta <= CLIP_MAX[args.dataset]
        # assert img[h2, w2, c2] - delta >= CLIP_MIN[args.dataset] and img[h2, w2, c2] + delta <= CLIP_MAX[args.dataset]
        n_step = 100 + 1
        tick = np.linspace(-delta, delta, n_step)
        losses = np.zeros([n_step, n_step])
        for j, d1 in enumerate(tqdm(tick)):
            x = np.tile(img, [n_step, 1, 1, 1])
            x[:, h1, w1, c1] += d1
            x[:, h2, w2, c2] += tick
            loss_line, = get_loss([x, keras.utils.to_categorical([label]*n_step, int(y_true.shape[1]))])
            losses[j, :] = loss_line

        X_grid, Y_grid = np.meshgrid(tick, tick)
        X_grid *= 2 / (CLIP_MAX[args.dataset] - CLIP_MIN[args.dataset])
        Y_grid *= 2 / (CLIP_MAX[args.dataset] - CLIP_MIN[args.dataset])
        surf = np.stack([X_grid, Y_grid, losses])
        np.save('vis/lossplot/surf_%s_%d.npy' % (args.dataset, i), surf)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X_grid, Y_grid, losses)
        plt.savefig('vis/lossplot/%s_%d_plot.png' % (args.dataset, i))
        plt.show()

        # X, Y, Z = surf.reshape([3, -1])
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot_trisurf(X, Y, Z)
        # plt.show()

        if args.dataset == 'imagenet':
            img = img[..., ::-1]
        imageio.imwrite('vis/lossplot/%s_%d_original.png' % (args.dataset, i), img)

        # return


def restore_surf():
    fl = glob.glob('vis/lossplot/surf_*.npy')
    for f in fl:
        surf = np.load(f)
        _, ds, id = f.split('_')  # 'cxr', '2.npy'
        id = id.split('.')[0]  # 2
        X_grid, Y_grid, losses = surf
        losses -= losses[len(losses)//2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X_grid, Y_grid, losses, cmap='jet', vmin=-0.02, vmax=0.02)
        ax.set_zlim(-0.02, 0.02)
        plt.savefig('vis/lossplot/%s_%s_plot.png' % (ds, id))
        plt.show()

        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use",
        required=True, type=str
    )
    parser.add_argument(
        '-a', '--attack',
        help="Attack to use train the discriminator; either 'fgsm', 'bim-a', 'bim-b', 'jsma', 'cw-l2'",
        required=False, type=str
    )


    # args = parser.parse_args()
    # analyze(args)

    for ds in ['imagenet', 'dr', 'cxr', 'derm']:
    #     for atk in ['fgsm', 'bim', 'pgd']:
            argv = ['-d', ds]
            print('\n$> ', argv)
            args = parser.parse_args(argv)
            analyze(args)


