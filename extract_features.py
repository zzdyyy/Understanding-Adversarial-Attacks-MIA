"""Extracting hidden layer features"""
from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
from datasets import get_data
from models import get_model
import keras
import numpy as np
import sklearn.metrics
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from global_config import *

DATASETS = ['dr', 'cxr', 'derm']
ATTACKS = ['fgsm', 'bim', 'jsma', 'cw-l2', 'clean']
TEST_SIZE = {'dr': 1.0, 'cxr': 1.0, 'derm': 1.0, 'imagenet': 1.0}
# TEST_SIZE = {'dr': 0.7, 'cxr': 0.7, 'derm': 0.6}

def extract(args):
    assert args.dataset in ['mnist', 'cifar-10', 'svhn', 'dr', 'cxr', 'derm', 'imagenet'], \
        "Dataset parameter must be either 'mnist', 'cifar-10', 'svhn', 'dr', 'cxr', or 'derm'"
    assert args.attack in ['clean', 'fgsm', 'bim', 'jsma', 'deepfool', 'pgd', 'ead', 'cw-l2', 'cw-lid', 'cw-li',
                           'fgsm_bb', 'bim_bb', 'jsma_bb', 'deepfool_bb', 'pgd_bb', 'ead_bb', 'cw-l2_bb', 'cw-lid_bb', 'cw-li_bb'], \
        "Attack parameter must be either 'fgsm', 'bim', 'jsma', 'deepfool', " \
        "'pgd', 'ead', 'cw-l2', 'cw-lid'"

    weights_file = "model/model_%s.h5" % args.dataset
    assert os.path.isfile(weights_file), \
        'model weights not found... must first train model using train_model.py.'

    print('Dataset: %s. Attack: %s' % (args.dataset, args.attack))

    model = get_model(args.dataset, softmax=True)
    model.load_weights(weights_file)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='sgd',
        metrics=['accuracy']
    )
    feat_model = keras.models.Model(model.input, model.layers[-1].input)

    if args.attack == 'clean':
        _, _, X_test, Y_test = get_data(args.dataset, split_traintest=False)

        feat = feat_model.predict(X_test, batch_size=args.batch_size, verbose=1)
        np.save('data/' + ADV_PREFIX + 'feat_%s_%s.npy' % (args.dataset, args.attack), feat)
        print('feature saved.')

        cnn_clf = keras.backend.function([model.layers[-1].input], [model.output])
        pred, = cnn_clf([feat])
        y_pred = pred.argmax(axis=1)
        y_true = Y_test.argmax(axis=1)
        idx_correct, = np.where(y_pred == y_true)
        print('find [%d] correct examples from total %d ones.'%(len(idx_correct), len(Y_test)))

        # split examples for train (the random svms) and testing (the detection)
        y_true = y_true[idx_correct]
        X_test = X_test[idx_correct]
        if TEST_SIZE[args.dataset]<1.0:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE[args.dataset], random_state=42)
            train_idx, test_idx = next(sss.split(X_test, y_true))
        else:
            train_idx = []
            test_idx = list(range(len(y_true)))
        print('Train: %d = %d negative + %d positive' % (len(train_idx), len(train_idx) - y_true[train_idx].sum(), y_true[train_idx].sum()))
        print('Test:  %d = %d negative + %d positive' % (len(test_idx), len(test_idx) - y_true[test_idx].sum(), y_true[test_idx].sum()))
        np.save('data/' + ADV_PREFIX + 'split_%s.npy' % args.dataset, (idx_correct, idx_correct[train_idx], idx_correct[test_idx]))

    else:
        if args.attack in ['cw-l2', 'cw-li', 'cw-lid']:
            save_file = 'data/' + ADV_PREFIX + 'Adv_%s_%s_%s.npy' % (args.dataset, args.attack, args.confidence)
        else:
            save_file = 'data/' + ADV_PREFIX + 'Adv_%s_%s.npy' % (args.dataset, args.attack)
        X_adv = np.load(save_file)
        feat = feat_model.predict(X_adv, batch_size=args.batch_size, verbose=1)
        np.save('data/' + ADV_PREFIX + 'feat_%s_%s.npy' % (args.dataset, args.attack), feat)
        print('feature saved.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use",
        required=True, type=str
    )
    parser.add_argument(
        '-a', '--attack',
        help="Attack to use train the discriminator; either 'clean' or 'fgsm', 'bim-a', 'bim-b', 'jsma', 'cw-l2'",
        required=True, type=str
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="batch size in computation",
        required=False, type=int
    )
    parser.add_argument(
        '-c', '--confidence',
        help="The confidence of the attack.",
        required=False, type=int
    )
    parser.set_defaults(batch_size=200)
    parser.set_defaults(confidence=200)


    args = parser.parse_args()
    extract(args)

    # for ds in ['dr', 'derm', 'cxr']:
    #     for atk in ['clean', 'fgsm', 'bim', 'pgd']:
    #         args = parser.parse_args(['-d', ds, '-a', atk, '-b', '200'])
    #         extract(args)

    # for ds in ['dr']:
    #     for atk in ['clean', 'fgsm', 'bim', 'pgd']:
    #         args = parser.parse_args(['-d', ds, '-a', atk, '-b', '200'])
    #         extract(args)

