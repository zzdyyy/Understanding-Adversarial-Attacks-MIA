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
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import random
from concurrent.futures import ThreadPoolExecutor

DATASETS = ['dr', 'cxr', 'derm']
ATTACKS = ['fgsm', 'bim', 'jsma', 'cw-l2', 'clean']


def balance_data(X, y, sub_sample=False):  # assume that the positive samples is less than negative ones
    idx_pos, = np.where(y>0)
    idx_neg, = np.where(y<1)
    if sub_sample:
        idx_neg = np.random.choice(idx_neg, len(idx_pos))
    else:  # over sample
        idx_pos = np.random.choice(idx_pos, len(idx_neg))
    idx_resample = np.concatenate([idx_neg, idx_pos])
    return X[idx_resample], y[idx_resample]


def get_cweights(dataset):  # get top layer linear-like classifier's weights, and the index from most to least important
    weights_file = 'data/cweights_%s.npy' % dataset
    if os.path.isfile(weights_file):
        print('[Cached weights] loading cweights from cache')
        w, idx = np.load(weights_file)
        idx = idx.astype(int)
        return w, idx
    else:
        import keras
        model = get_model(dataset)
        w = model.layers[-1].weights[0].eval(keras.backend.get_session())
        w = np.abs(w[:, 1] - w[:, 0])
        idx = np.argsort(w)[::-1]
        np.save(weights_file, (w, idx))
        return w, idx


def get_random_svms(dataset, attack, *cfg):
    from sklearn.externals import joblib
    svm_file = 'data/ensamblesvm_%s_%s.model' % (dataset, attack)
    if os.path.isfile(svm_file):
        print('[Cached rsvms] loading random SVMs from cache')
        rsvms = joblib.load(svm_file)
        return rsvms
    else:
        rsvms = train_random_svms(*cfg)
        joblib.dump(rsvms, svm_file)
        return rsvms


def train_random_svms(important_ratio, svm_dim, svm_num, svm_imp_ratio, X_train, isadv_train):
    # dimensional importance
    w, idx = get_cweights(args.dataset)
    imp_line = int(len(idx) * important_ratio)
    imp_idx = idx[:imp_line]
    unimp_idx = idx[imp_line:]

    def train_worker(i):
        print('[svm %d] start training svm ...' % i)
        imp_num = int(svm_dim * svm_imp_ratio)
        f_idx = np.concatenate([
            np.random.choice(imp_idx, imp_num),
            np.random.choice(unimp_idx, svm_dim - imp_num)
            ])
        X = X_train[:, f_idx]
        rescale = np.sum(w) / np.sum(w[f_idx])
        X *= rescale
        svm = sklearn.svm.SVC(kernel='linear', probability=True)
        svm.fit(X, isadv_train)
        print('[svm %d] training over.' % i)
        print('[svm %d] accuracy on train:' % i, sklearn.metrics.accuracy_score(isadv_train, svm.predict(X)))
        return f_idx, rescale, svm

    with ThreadPoolExecutor(10) as executor:
        rsvms = executor.map(train_worker, range(svm_num))
    return list(rsvms)


def detect(args):
    assert args.dataset in ['mnist', 'cifar-10', 'svhn', 'dr', 'cxr', 'derm'], \
        "Dataset parameter must be either 'mnist', 'cifar-10', 'svhn', 'dr', 'cxr', or 'derm'"
    assert args.attack in ['fgsm', 'bim', 'jsma', 'deepfool', 'pgd', 'ead', 'cw-l2', 'cw-lid'], \
        "Attack parameter must be either 'fgsm', 'bim', 'jsma', 'deepfool', " \
        "'pgd', 'ead', 'cw-l2', 'cw-lid'"

    important_ratio = 0.5  # how many dims are important
    svm_dim = 30  # the dims used by a svm
    svm_num = 10  # how many svms
    svm_imp_ratio = 0.1  # how many important dims in 'svm_dim' dims

    # load training/testing data
    cX_train, cy_train, cX_test, cy_test = get_data(args.dataset, onehot=False, load_feat='clean')  # clean feat
    aX_train, ay_train, aX_test, ay_test = get_data(args.dataset, onehot=False, load_feat=args.attack)  # attack feat

    # prepare data for training svms
    cX_train, cy_train = balance_data(cX_train, cy_train)  # balance over positive/negative examples
    aX_train, ay_train = balance_data(aX_train, ay_train)  # balance over positive/negative examples
    X_train = np.concatenate([cX_train, aX_train])
    isadv_train = np.concatenate([np.zeros(len(cX_train)), np.ones(len(aX_train))])

    # do PCA
    do_pca = True
    if do_pca:
        pca = PCA(20, True)
        X_pca = pca.fit_transform(X_train)
        print('PCA explanation of variances:', (pca.explained_variance_ratio_))
        plt.scatter(X_pca[isadv_train<1, 0], X_pca[isadv_train<1, 1])
        plt.scatter(X_pca[isadv_train>0, 0], X_pca[isadv_train>0, 1])
        plt.show()

    rsvms = get_random_svms(args.dataset, args.attack, important_ratio, svm_dim, svm_num, svm_imp_ratio, X_train, isadv_train)

    # prepare data for testing
    cX_test, cy_test = balance_data(cX_test, cy_test)  # balance over positive/negative examples
    aX_test, ay_test = balance_data(aX_test, ay_test)  # balance over positive/negative examples
    X_test = np.concatenate([cX_test, aX_test])
    isadv_test = np.concatenate([np.zeros(len(cX_test)), np.ones(len(aX_test))])

    def svm_predict(pargs):
        f_idx, rescale, svm = pargs
        p = svm.predict_proba(X_test[:, f_idx] * rescale)[:, 1:]
        print('svm test accuracy: ', accuracy_score(isadv_test, (p>0.5).astype('float')))
        return p
    with ThreadPoolExecutor(10) as executor:
        probs = executor.map(svm_predict, rsvms)
    probs = list(probs)  # probs of different svm
    probs = np.concatenate(probs, axis=1)
    X = probs
    y = isadv_test

    # do PCA
    do_pca = True
    if do_pca:
        pca = PCA(10, True)
        X_pca = X  # pca.fit_transform(X)
        #print('PCA explanation of variances:', (pca.explained_variance_ratio_))
        plt.scatter(X_pca[y < 1, 0], X_pca[y < 1, 1])
        plt.scatter(X_pca[y > 0, 0], X_pca[y > 0, 1])
        plt.show()

    # split testing data to detection train/test set
    dX_train, dX_test, dy_train, dy_test = train_test_split(X, y, test_size=0.5)
    detector = SVC(gamma=2.8, probability=True)
    detector.fit(dX_train, dy_train)
    print('Accuracy: ', accuracy_score(dy_test, detector.predict(dX_test)))
    print('ROC:', roc_auc_score(dy_test, detector.predict_proba(dX_test)[:, 1]))

    print('end')


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
        required=True, type=str
    )


    args = parser.parse_args()
    detect(args)

    # for ds in ['derm', 'dr', 'cxr']:
    #     for atk in ['fgsm', 'bim', 'pgd']:
    #         args = parser.parse_args(['-d', ds, '-a', atk])
    #         detect(args)

