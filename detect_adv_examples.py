from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import numpy as np
from sklearn.preprocessing import scale, MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from util import (random_split, block_split, train_lr, compute_roc)
from sklearn.metrics import roc_curve, auc, roc_auc_score
from global_config import *
from sklearn.model_selection import train_test_split
from datasets import get_data

DATASETS = ['mnist', 'cifar-10', 'svhn', 'dr', 'cxr', 'derm']
ATTACKS = ['fgsm', 'bim-a', 'bim-b', 'jsma', 'cw-l2']
CHARACTERISTICS = ['kd', 'bu', 'lid']
PATH_DATA = "data/"
PATH_IMAGES = "plots/"

def load_characteristics(dataset, attack, characteristics, k, q):
    """
    Load multiple characteristics for one dataset and one attack.
    k/q are for LIDq estimation, different setttings
    """
    X, Y = None, None
    for ch in characteristics:
        # print("  -- %s" % characteristics)
        if ch == 'lid':
            file_name = os.path.join(PATH_DATA, ADV_PREFIX + "%s_%s_%s_%s_%s.npy" % (ch, dataset, attack, k, q))
        else:
            file_name = os.path.join(PATH_DATA, ADV_PREFIX + "%s_%s_%s.npy" % (ch, dataset, attack))
        data = np.load(file_name)
        if X is None:
            X = data[:, :-1]
        else:
            X = np.concatenate((X, data[:, :-1]), axis=1)
        if Y is None:
            Y = data[:, -1] # labels only need to load once

    return X, Y

def balance_data(X, Y, y, sub_sample=False):  # assume that the positive samples is less than negative ones
    idx_pos, = np.where(y>0)
    idx_neg, = np.where(y<1)
    if sub_sample:
        idx_neg = np.random.choice(idx_neg, len(idx_pos))
    else:  # over sample
        idx_pos = np.random.choice(idx_pos, len(idx_neg))
    idx_resample = np.concatenate([idx_neg, idx_pos])
    return X[idx_resample], Y[idx_resample], y[idx_resample]

def detect(args):
    chars = args.characteristics.split(',')
    print("Loading train attack: %s" % args.attack)
    X, Y = load_characteristics(args.dataset, args.attack, chars, args.lid_k, args.lid_q)

    balance = True
    if balance:
        _, _, _, cy_test = get_data(args.dataset, onehot=False, load_feat='clean')
        X, Y, _ = balance_data(X, Y, np.tile(cy_test, 2))

    # standarization
    scaler = MinMaxScaler().fit(X)
    X = scaler.transform(X)
    # X = scale(X) # Z-norm

    accs = []
    aucs = []
    for i in range(3):
        # test attack is the same as training attack
        # X_train, Y_train, X_test, Y_test = block_split(X, Y)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        if args.test_attack is not None and args.test_attack != args.attack:
            # test attack is a different attack
            print("Loading test attack: %s" % args.test_attack)
            X_test, Y_test = load_characteristics(args.dataset, args.test_attack, chars, args.lid_k, args.lid_q)
            _, _, X_test, Y_test = block_split(X_test, Y_test)

            # apply training normalizer
            X_test = scaler.transform(X_test)
            # X_test = scale(X_test) # Z-norm

        print("Train data size: ", X_train.shape)
        print("Test data size: ", X_test.shape)


        ## Build detector
        print("LR Detector on [dataset: %s, train_attack: %s, test_attack: %s] with:" %
                                            (args.dataset, args.attack, args.test_attack))
        lr = train_lr(X_train, Y_train)

        ## Evaluate detector
        y_pred = lr.predict_proba(X_test)[:, 1]
        y_label_pred = lr.predict(X_test)

        # compute scores
        # fpr, tpr, _ = roc_curve(Y_test, y_pred)
        auc_score = roc_auc_score(Y_test, y_pred)
        aucs.append(auc_score)
        precision = precision_score(Y_test, y_label_pred)
        recall = recall_score(Y_test, y_label_pred)

        y_label_pred = lr.predict(X_test)
        acc = accuracy_score(Y_test, y_label_pred)
        accs.append(acc)
        print('=================')
        print('k=%s, q=%s' % (args.lid_k, args.lid_q))
        print('Detector ROC-AUC score: %0.4f, accuracy: %.4f, precision: %.4f, recall: %.4f' %
              (auc_score, acc, precision, recall))
        print('=================')

    log = 'Dataset: %5s, Attack: %10s, Acc: %.4f, AUC: %.4f, Accs: %s, AUCs: %s \n' % \
          (args.dataset, args.attack, np.mean(accs), np.mean(aucs), str(accs), str(aucs))
    print(log)
    with open('log/detect_%s.log' % args.characteristics, 'a') as f:
        f.write(log)


    return lr, auc_score, scaler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar-10' or 'svhn'",
        required=True, type=str
    )
    parser.add_argument(
        '-a', '--attack',
        help="Attack to use train the discriminator; either 'fgsm', 'bim-a', 'bim-b', 'jsma' 'cw-l2'",
        required=True, type=str
    )
    parser.add_argument(
        '-r', '--characteristics',
        help="Characteristic(s) to use any combination in ['kd', 'bu', 'lid'] "
             "separated by comma, for example: kd,bu",
        required=True, type=str
    )
    parser.add_argument(
        '-t', '--test_attack',
        help="Characteristic(s) to cross-test the discriminator.",
        required=False, type=str
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False, type=int
    )
    parser.add_argument(
        '-k', '--lid_k',
        help="The number of nearest neighbours to use; either 10, 20, 100 ",
        required=False, type=int
    )
    parser.add_argument(
        '-q', '--lid_q',
        help="The q parameter for LIDq estimation ",
        required=False, type=float
    )
    parser.set_defaults(batch_size=100)
    parser.set_defaults(test_attack=None)
    parser.set_defaults(lid_k=20)
    parser.set_defaults(lid_q=1.0)
    # args = parser.parse_args()
    # detect(args)

    for ch in ['lid', 'kd', 'bu']:
        for ds in ['dr', 'cxr', 'derm']:
            for atk in ['cw-li']:  # 'fgsm', 'bim', 'deepfool', 'pgd']:
                argv = ['-d', ds, '-a', atk, '-r', ch]
                print('\n$> ', argv)
                args = parser.parse_args(argv)
                detect(args)

    # for k in [30, 50, 100]:
    #     for q in [1.0, 2.0]:
    #         argv = ['-d', 'derm', '-a', 'fgsm', '-t', 'fgsm', '-r', 'lid',
    #                                   '-k', str(k), '-q', str(q)]
    #         print(argv)
    #         args = parser.parse_args(argv)
    #         detect(args)
