from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import warnings
import numpy as np
import tensorflow as tf
from keras.models import load_model
import sklearn.metrics

from models import get_model
from datasets import get_data

from cleverhans.evaluation import batch_eval
from cleverhans.attacks import FastGradientMethod, BasicIterativeMethod, \
    SaliencyMapMethod, CarliniWagnerL2, ElasticNetMethod, DeepFool, LBFGS, \
    MadryEtAl
from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import KerasModelWrapper
from cw_attacks import CarliniL2, CarliniLID
from pgd_attack import LinfPGDAttack
from ead_attack import EADL1
from global_config import *

import keras
import keras.backend as K
K.set_learning_phase(0)

# attack settings that were chosen
# theta=1 means, the pixel value xi will be changed to xi=xi+theta (-0.5 will be changed to 1)
# gamma means the percentage of pixels changed
eps = {'dr': 2/255*2, 'cxr': 3/255*2, 'derm': 2/255*2, 'imagenet': 1/255*255}
SETTING = {
    'mnist': {'eps': 0.3, 'eps_iter': 0.01, 'nb_iter': 40,
              'theta': 1., 'gamma': 0.145, 'image_size': 28,
              'num_channels': 1, 'num_labels': 10},
    'cifar-10': {'eps': 0.031, 'eps_iter': 0.003, 'nb_iter': 20,
                 'theta': 1., 'gamma': 0.145, 'image_size': 32,
                 'num_channels': 3, 'num_labels': 10},
    'svhn': {'eps': 0.05, 'eps_iter': 0.005, 'nb_iter': 20,
             'theta': 1., 'gamma': 0.145, 'image_size': 32,
             'num_channels': 3, 'num_labels': 10},
    'dr':   {'eps': eps['dr'],   'eps_iter': eps['dr']/20,   'nb_iter': 20, 'theta': 1., 'gamma': 0.145, 'image_size': 224, 'num_channels': 3, 'num_labels': 2},  # TODO: not determined
    'cxr':  {'eps': eps['cxr'],  'eps_iter': eps['cxr']/20,  'nb_iter': 20, 'theta': 1., 'gamma': 0.145, 'image_size': 224, 'num_channels': 3, 'num_labels': 2},
    'derm': {'eps': eps['derm'], 'eps_iter': eps['derm']/20, 'nb_iter': 20, 'theta': 1., 'gamma': 0.145, 'image_size': 224, 'num_channels': 3, 'num_labels': 2},
    'imagenet': {'eps': eps['derm'], 'eps_iter': eps['derm']/20, 'nb_iter': 20, 'theta': 1., 'gamma': 0.145, 'image_size': 224, 'num_channels': 3, 'num_labels': 2},
}  # eps_iter is for BIM, and it will be doubled before applying deepfool

CLIP_MIN = {'mnist': -0.5, 'cifar': -0.5, 'svhn': -0.5, 'dr': -1.0, 'cxr': -1.0, 'derm': -1.0, 'imagenet':-128.0}
CLIP_MAX = {'mnist':  0.5, 'cifar':  0.5, 'svhn':  0.5, 'dr':  1.0, 'cxr':  1.0, 'derm':  1.0, 'imagenet':128.0}



def main(args):
    assert args.dataset in ['mnist', 'cifar-10', 'svhn', 'dr', 'cxr', 'derm', 'imagenet'], \
        "Dataset parameter must be either 'mnist', 'cifar-10', 'svhn', 'dr', 'cxr', or 'derm'"
    assert args.attack in ['fgsm', 'bim', 'jsma', 'deepfool', 'pgd', 'ead', 'cw-l2', 'cw-lid'], \
        "Attack parameter must be either 'fgsm', 'bim', 'jsma', 'deepfool', " \
        "'pgd', 'ead', 'cw-l2', 'cw-lid'"

    if args.epsilon:
        SETTING[args.dataset]['eps'] = args.epsilon/255*(CLIP_MAX[args.dataset] - CLIP_MIN[args.dataset])
        SETTING[args.dataset]['eps_iter'] = SETTING[args.dataset]['eps'] / SETTING[args.dataset]['nb_iter']

    weights_file = "model/model_%s%s.h5" % (args.dataset, '_bb' if args.blackbox else '')
    assert os.path.isfile(weights_file), \
        'model weights not found... must first train model using train_model.py.'

    print('Dataset: %s. Attack: %s, Confidence: %s' % (args.dataset, args.attack, args.confidence))
    _, _, X_test, Y_test = get_data(args.dataset, split_traintest=False)
    # specify the number of images to attack
    X_test = X_test[:args.num_attack]
    Y_test = Y_test[:args.num_attack]

    if args.dataset == 'imagenet':
        CLIP_MAX[args.dataset] = float(X_test.max())
        CLIP_MIN[args.dataset] = float(X_test.min())

    n_images = X_test.shape[0]
    image_shape = X_test.shape[1:]
    n_class = Y_test.shape[1]

    # Create TF session, set it as Keras backend
    sess = tf.Session()
    K.set_session(sess)


    # Load one specific attack type
    if args.attack in ['cw-l2', 'cw-li', 'cw-lid']:
        save_file = 'data/' + ADV_PREFIX + 'Adv_%s_%s%s_%s.npy' % (args.dataset, args.attack,
                                                                   '_bb' if args.blackbox else '', args.confidence)
    else:
        save_file = 'data/' + ADV_PREFIX + 'Adv_%s_%s%s.npy' % (args.dataset, args.attack,
                                                                '_bb' if args.blackbox else '')
    X_adv = np.load(save_file)
    print('Adversarial samples loaded from %s ' % save_file)
    log = '%40s:' % save_file

    # evaluate the attacks and model
    model = get_model(args.dataset, softmax=True)
    weights_file = "model/model_%s.h5" % args.dataset
    model.load_weights(weights_file)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    y_pred = model.predict_classes(X_test, verbose=1, batch_size=args.batch_size)
    y_true = Y_test.argmax(-1)
    acc = sklearn.metrics.accuracy_score(y_true, y_pred)
    auroc = sklearn.metrics.roc_auc_score(y_true, y_pred)
    print("Accuracy on clean test set: %0.2f%%" % (100*acc))
    log += '\t' + "Accuracy on clean test set: %0.2f%%" % (100*acc)
    print('AUROC on clean: %0.4f' % auroc)
    log += '\t' + 'AUROC on clean: %0.4f' % auroc

    # statistics of the attacks
    idx_correct, = np.where(y_pred == y_true)
    y_pred = model.predict_classes(X_adv, verbose=1, batch_size=args.batch_size)
    acc_1 = sklearn.metrics.accuracy_score(y_true, y_pred)
    acc_2 = sklearn.metrics.accuracy_score(y_true[idx_correct], y_pred[idx_correct])
    auroc = np.nan
    if args.dataset != 'imagenet':
        auroc = sklearn.metrics.roc_auc_score(y_true[idx_correct], y_pred[idx_correct])
    print("Model accuracy on the adversarial test set: %0.2f%% %0.2f%%" % (100 * acc_1, 100 * acc_2))
    log += '\t' + "Model accuracy on the adversarial test set: %0.2f%% %0.2f%%" % (100 * acc_1, 100 * acc_2)
    print('AUROC on adv: %0.4f' % auroc)
    log += '\t' + 'AUROC on adv: %0.4f' % auroc

    l2_diff = np.linalg.norm(
        X_adv.reshape((len(X_test), -1)) -
        X_test.reshape((len(X_test), -1)),
        axis=1
    ).mean()
    print("Average L-2 distortion [%s : %0.4f]" % (args.attack, l2_diff))
    log += '\t' + "Average L-2 distortion [%s : %0.4f]" % (args.attack, l2_diff)

    li_diff = np.max(
        np.abs(
            X_adv.reshape((len(X_test), -1)) -
            X_test.reshape((len(X_test), -1))),
        axis=0
    ).mean()
    print("Average L-i distortion [%s : %0.4f]" % (args.attack, li_diff))
    log += '\t' + "Average L-i distortion [%s : %0.4f]" % (args.attack, li_diff)

    log += '\t' + str(SETTING[args.dataset]) + str(args) + '\n'
    with open('log/%s.log' % args.dataset, 'a') as f:
        f.write(log)

    sess.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar-10', 'svhn', 'dr', 'cxr', or 'derm'",
        required=True, type=str
    )
    parser.add_argument(
        '-a', '--attack',
        help="Attack to use; either 'fgsm', 'bim', 'jsma', "
             "'deepfool', 'pgd', 'ead', 'cw-l2', 'cw-lid' ",
        required=True, type=str
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False, type=int
    )
    parser.add_argument(
        '-c', '--confidence',
        help="The confidence of the attack.",
        required=False, type=int
    )
    parser.add_argument(
        '-n', '--num_attack',
        help="The number of attack to craft.",
        required=False, type=int
    )
    parser.add_argument(
        '-e', '--epsilon',
        help="The maximum L-i of perturbation.",
        required=False, type=float
    )
    parser.add_argument(
        '--blackbox',
        help="Load blackbox model to craft black box attack.",
        required=False, type=bool, default=False
    )
    parser.set_defaults(batch_size=100)
    parser.set_defaults(confidence=0)
    parser.set_defaults(num_attack=20000)
    args = parser.parse_args()

    main(args)
    #
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    #
    # for dataset in ['cifar-10']:
    #     for attack in ['fgsm', 'pgd']:
    #         args = parser.parse_args(['-d', dataset, '-a', attack, '-b', '100'])
    #         main(args)
    #
    # K.clear_session()
