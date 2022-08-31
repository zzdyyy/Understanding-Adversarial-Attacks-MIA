"""Detect adv/clean from the hidden feature"""
from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
from datasets import get_data
from models import get_model
import numpy as np
import sklearn.metrics
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import (SVC, LinearSVC)
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

DATASETS = ['dr', 'cxr', 'derm']
ATTACKS = ['fgsm', 'bim', 'jsma', 'cw-l2', 'clean']
TEST_SIZE = {'dr': 0.0, 'cxr': 0.0, 'derm': 0.0}

def balance_data(X, y, sub_sample=False):  # assume that the positive samples is less than negative ones
    idx_pos, = np.where(y>0)
    idx_neg, = np.where(y<1)
    if sub_sample:
        idx_neg = np.random.choice(idx_neg, len(idx_pos))
    else:  # over sample
        idx_pos = np.random.choice(idx_pos, len(idx_neg))
    idx_resample = np.concatenate([idx_neg, idx_pos])
    return X[idx_resample], y[idx_resample]


def detect(args):
    assert args.dataset in ['mnist', 'cifar-10', 'svhn', 'dr', 'cxr', 'derm'], \
        "Dataset parameter must be either 'mnist', 'cifar-10', 'svhn', 'dr', 'cxr', or 'derm'"

    attacks_src = ['fgsm', 'pgd']
    attacks = ['fgsm', 'bim', 'pgd', 'cw-li']

    # load feature/label data and balance data
    # clean examples
    _, _, cX_test, cy_test = get_data(args.dataset, onehot=False, load_feat='clean')  # clean feat
    cX_test, cy_test = balance_data(cX_test, cy_test)  # balance over positive/negative examples
    # adv examples
    aX, ay = [], []
    for i, attack in enumerate(attacks):
        _, _, aX_test, ay_test = get_data(args.dataset, onehot=False, load_feat=attack)  # attack feat
        aX_test, ay_test = balance_data(aX_test, ay_test)  # balance over positive/negative examples
        aX.append(aX_test)
        ay.append(ay_test)

    auc_matrix = np.zeros([len(attacks_src), len(attacks)])
    for aid_train in range(len(attacks_src)):  # attack ID for training
        print('training on attack:', attacks_src[aid_train])

        X = np.concatenate([cX_test, aX[aid_train]])
        y = np.concatenate([np.zeros(len(cX_test)), np.ones(len(aX[aid_train]))])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE[args.dataset])
        X_test, y_test = X, y
        clf = SVC(gamma=2.8, probability=True)  #RandomForestClassifier(30)  # SVC(gamma=2.8, probability=True)
        clf.fit(X_train, y_train)
        # auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        # print('AUC:', auc)
        # auc_matrix[aid_train, aid_train] = auc

        for aid_test in range(len(attacks)):
            # if aid_train == aid_test:
            #     continue

            X_test = np.concatenate([cX_test, aX[aid_test]])
            y_test = np.concatenate([np.zeros(len(cX_test)), np.ones(len(aX[aid_test]))])

            auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
            print('AUC:', auc, '(test on', attacks[aid_test]+')')
            auc_matrix[aid_train, aid_test] = auc


    print(auc_matrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use",
        required=True, type=str
    )
    # parser.add_argument(
    #     '-a', '--attack',
    #     help="Attack to use train the discriminator; either 'fgsm', 'bim-a', 'bim-b', 'jsma', 'cw-l2'",
    #     required=True, type=str
    # )


    args = parser.parse_args()
    detect(args)

    # with ThreadPoolExecutor(24) as e:
    #     for ds in ['dr', 'cxr', 'derm']:
    #         for atk in ['pgd']:  # 'fgsm', 'bim', 'deepfool', 'pgd']:
    #             argv = ['-d', ds, '-a', atk]
    #             print('\n$> ', argv)
    #             args = parser.parse_args(argv)
    #             e.submit(detect, args) #detect(args)
    #     e.result()

