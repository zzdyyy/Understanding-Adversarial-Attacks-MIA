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
from global_config import *
import glob

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

    def reg(x):
        if x.shape[-1] == 1:
            x = np.tile(x, [1, 1, 3])
        return (x - x.min()) / (x.max() - x.min())

    label_to_save = {
        'derm': -1,
        'imagenet': 2,
        'dr' : -6,
        'cxr': 0,
        #'mnist': np.where(y == 6)[0][1],
    }

    # img, label = X[label_to_save[args.dataset]], y[label_to_save[args.dataset]]
    # grads = visualize_saliency(model, layer_idx, filter_indices=label, seed_input=img)
    # cgrads = cv2.GaussianBlur(grads, (51,51), 3)
    # cgrads = reg(cgrads)
    # cgrads = cv2.applyColorMap(((1 - cgrads) * 255.9).astype('uint8'), cv2.COLORMAP_JET)
    # imageio.imsave('vis/saliency/%s_one%d_saliency.png' % (args.dataset, label_to_save[args.dataset]), cgrads)
    # imageio.imsave('vis/saliency/%s_one%d_original.png' % (args.dataset, label_to_save[args.dataset]), reg(img))

    plot_range = {
        'imagenet': slice(0, None),  # 2
        'derm': np.concatenate([ np.where(y>0)[0][-5:], np.where(y<1)[0][:5]  ]),   # -1
        'dr': np.concatenate([ np.where(y>0)[0][-5:], np.where(y<1)[0][:5]  ]),   # -6
        'cxr': np.concatenate([ np.where(y>0)[0][-5:], np.where(y<1)[0][:5]  ]),   #-
        'mnist': np.where(y == 6)[0][:10],
    }

    # for i, (img, label) in enumerate(zip(X[plot_range[args.dataset]], y[plot_range[args.dataset]])):
    #     grads = visualize_saliency(model, layer_idx, filter_indices=label, seed_input=img)
    #     # grads = cv2.GaussianBlur(grads, (51, 51), 3 if args.dataset != 'mnist' else 0.8)
    #     grads = reg(grads)
    #     grads = cv2.applyColorMap(((1 - grads) * 255.9).astype('uint8'), cv2.COLORMAP_JET)
    #     if args.dataset == 'imagenet':
    #         img = img[..., ::-1]
    #
    #     imageio.imsave('vis/saliency/%s_%d_saliency.png' % (args.dataset, i), grads)
    #     imageio.imsave('vis/saliency/%s_%d_original.png' % (args.dataset, i), reg(img))
    #
    #     plt.imshow(reg(grads));plt.show()
    #     plt.imshow(reg(img));plt.show()


    for i, (img, label) in enumerate(zip(X[plot_range[args.dataset]], y[plot_range[args.dataset]])):
        grads, = get_grad([img[None, ...], keras.utils.to_categorical(label, int(y_true.shape[1]))])
        grads = np.sum(grads[0], axis=2)
        grads = grads * (CLIP_MAX[args.dataset] - CLIP_MIN[args.dataset])
        # grads = cv2.GaussianBlur(grads, (51, 51), 1 if args.dataset != 'mnist' else 0.8)
        if args.dataset == 'imagenet':
            img = img[..., ::-1]
        # plt.hist(np.abs(grads).reshape([-1]), bins=200);plt.show()
        plt.imshow(np.abs(grads), cmap='jet', vmin=0, vmax=0.2);plt.colorbar();plt.axis('off');plt.savefig('vis/saliency/%s_%d_saliency.png' % (args.dataset, i));plt.show()
        plt.imshow(reg(img));imageio.imwrite('vis/saliency/%s_%d_original.png' % (args.dataset, i), reg(img));plt.show()

        # grads = cv2.GaussianBlur(grads, (51, 51), 3 if args.dataset != 'mnist' else 0.8)
        # grads = reg(grads)
        # grads = cv2.applyColorMap(((1 - grads) * 255.9).astype('uint8'), cv2.COLORMAP_JET)

        # imageio.imsave('vis/saliency/%s_%d_saliency.png' % (args.dataset, i), grads)
        # imageio.imsave('vis/saliency/%s_%d_original.png' % (args.dataset, i), reg(img))



    # grads, = batch_eval(keras.backend.get_session(), [x_in, y_true], [x_grad], [X[:100], keras.utils.to_categorical(y[:100], int(y_true.shape[1]))], 10)
    # gs = (grads).reshape([-1])
    # #gs = np.tile(gs, 10);    gs = 128 * gs
    # plt.hist(gs, bins=500)#, range=(-1, 1));
    # plt.yscale('log')
    # plt.show()
    # iutils.postprocess_images()


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


