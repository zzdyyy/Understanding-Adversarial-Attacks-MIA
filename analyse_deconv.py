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
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from global_config import *
import imageio

import deeplift
import innvestigate
import innvestigate.utils as iutils
import innvestigate.utils.visualizations as ivis
import eutils

DATASETS = ['dr', 'cxr', 'derm']
ATTACKS = ['fgsm', 'bim', 'jsma', 'cw-l2', 'clean']
TEST_SIZE = {'dr': 0.7, 'cxr': 0.7, 'derm': 0.5}

CLIP_MIN = {'mnist': -0.5, 'cifar': -0.5, 'svhn': -0.5, 'dr': -1.0, 'cxr': -1.0, 'derm': -1.0}
CLIP_MAX = {'mnist':  0.5, 'cifar':  0.5, 'svhn':  0.5, 'dr':  1.0, 'cxr':  1.0, 'derm':  1.0}


def balance_data(X, y, sub_sample=False):  # assume that the positive samples is less than negative ones
    idx_pos, = np.where(y>0)
    idx_neg, = np.where(y<1)
    if sub_sample:
        idx_neg = np.random.choice(idx_neg, len(idx_pos))
    else:  # over sample
        idx_pos = np.random.choice(idx_pos, len(idx_neg))
    idx_resample = np.concatenate([idx_neg, idx_pos])
    return X[idx_resample], y[idx_resample]


def bk_proj(X):
    return ivis.graymap(X)


def heatmap(X):
    return ivis.heatmap(X)


def graymap(X):
    return ivis.graymap(np.abs(X)) #, input_is_postive_only=True)


def solve_name_controdiction(model):
    for name in map(lambda x: x.__class__.__name__, model.layers):
        K.get_uid(name)
    K.get_uid('input')
    K.get_uid('input')

def detect(args):
    assert args.dataset in ['mnist', 'cifar-10', 'svhn', 'dr', 'cxr', 'derm'], \
        "Dataset parameter must be either 'mnist', 'cifar-10', 'svhn', 'dr', 'cxr', or 'derm'"
    assert args.attack in ['fgsm', 'bim', 'jsma', 'deepfool', 'pgd', 'ead', 'cw-l2', 'cw-lid'], \
        "Attack parameter must be either 'fgsm', 'bim', 'jsma', 'deepfool', " \
        "'pgd', 'ead', 'cw-l2', 'cw-lid'"

    # load feature/label data
    cX_train, cy_train, cX_test, cy_test = get_data(args.dataset, onehot=False)  # clean feat
    _, _, aX_test, ay_test = get_data(args.dataset, onehot=False)  # attack feat

    correct_idx, train_idx, test_idx = np.load('data/' + ADV_PREFIX + 'split_%s.npy' % args.dataset, allow_pickle=True)
    aX_all = np.load('data/' + ADV_PREFIX + 'Adv_%s_%s.npy' % (args.dataset, args.attack))
    aX_test = aX_all[test_idx]

    image_shape = cX_test.shape[1:]


    model = get_model(args.dataset)
    solve_name_controdiction(model)


    input_range = [CLIP_MIN[args.dataset], CLIP_MAX[args.dataset]]
    # Scale to [0, 1] range for plotting.
    def input_postprocessing(X):
        return (X - CLIP_MIN[args.dataset]) / (CLIP_MAX[args.dataset] - CLIP_MIN[args.dataset])

    noise_scale = (input_range[1] - input_range[0]) * 0.1
    ri = input_range[0]  # reference input

    methods = [
        # NAME                    OPT.PARAMS                POSTPROC FXN               TITLE

        # Show input
        ("input",                 {},                       input_postprocessing,      "Input"),

        # Function
        ("gradient",              {"postprocess": "abs"},   graymap,        "Gradient"),
        # ("smoothgrad",            {"noise_scale": noise_scale,
        #                            "postprocess": "square"},graymap,        "SmoothGrad"),

        # Signal
        ("deconvnet",             {},                       bk_proj,        "Deconvnet"),
        ("guided_backprop",       {},                       bk_proj,        "Guided Backprop",),
        #("pattern.net",           {"pattern_type": "relu"}, bk_proj,        "PatternNet"),

        # Interaction
        #("pattern.attribution",   {"pattern_type": "relu"}, heatmap,        "PatternAttribution"),
        ("deep_taylor.bounded",   {"low": input_range[0],
                                   "high": input_range[1]}, heatmap,        "DeepTaylor"),
        ("input_t_gradient",      {},                       heatmap,        "Input * Gradient"),
        # ("integrated_gradients",  {"reference_inputs": ri}, heatmap,        "Integrated Gradients"),
        # ("deep_lift.wrapper",     {"reference_inputs": ri}, heatmap,        "DeepLIFT Wrapper - Rescale"),
        # ("deep_lift.wrapper",     {"reference_inputs": ri, "nonlinear_mode": "reveal_cancel"},
        #                                                     heatmap,        "DeepLIFT Wrapper - RevealCancel"),
        ("lrp.z",                 {},                       heatmap,        "LRP-Z"),
        ("lrp.epsilon",           {"epsilon": 1},           heatmap,        "LRP-Epsilon"),
    ]

    model_wo_softmax = get_model(args.dataset, softmax=False)
    analyzers = []
    for method in methods:
        analyzer = innvestigate.create_analyzer(method[0],  # analysis method identifier
                                                model_wo_softmax,  # model without softmax output
                                                **method[1])  # optional analysis parameters
        # Some analyzers require training.
        analyzer.fit(cX_train, batch_size=256, verbose=1)
        analyzers.append(analyzer)

    idx_pos, = np.where(cy_test>0)
    idx_neg, = np.where(cy_test<1)


    def analyse(test_images, output_path):
        analysis = np.zeros([len(test_images), len(analyzers)] + list(image_shape))
        text = []

        for i, (x, y) in enumerate(test_images):
            # Add batch axis.
            x = x[None, :, :, :]

            # Predict final activations, probabilites, and label.
            presm = model_wo_softmax.predict_on_batch(x)[0]
            prob = model.predict_on_batch(x)[0]
            y_hat = prob.argmax()

            # Save prediction info:
            text.append(("%s" % str(y),  # ground truth label
                         "%.2f" % presm.max(),  # pre-softmax logits
                         "%.2f" % prob.max(),  # probabilistic softmax output
                         "%s" % str(y_hat) # predicted label
                         ))

            for aidx, analyzer in enumerate(analyzers):
                # Analyze.
                print('analyzing', analyzer, '...')
                a = analyzer.analyze(x)

                # Apply common postprocessing, e.g., re-ordering the channels for plotting.
                # a = postprocess(a)
                # Apply analysis postprocessing, e.g., creating a heatmap.
                a = methods[aidx][2](a)
                # Store the analysis.
                analysis[i, aidx] = a[0]

        # analysis_c = np.concatenate(np.concatenate(analysis, axis=2), axis=0)
        analysis_c = analysis[:, -2, ...]  # lrp.z
        analysis_c = np.reshape(analysis_c, [6, 6, 224, 224, 3])
        analysis_c = np.concatenate(np.concatenate(analysis_c, axis=2), axis=0)
        imageio.imsave(output_path, analysis_c)
        return analysis

    n = 36
    ana1 = analyse(list(zip(cX_test[idx_pos][:n], cy_test[idx_pos][:n])), 'vis/ana_cpos.png')
    # analyse(list(zip(cX_test[idx_neg][:n], cy_test[idx_neg][:n])), 'vis/ana_cneg.png')
    # analyse(list(zip(aX_test[idx_pos][:n], cy_test[idx_pos][:n])), 'vis/ana_apos.png')
    ana2 = analyse(list(zip(aX_test[idx_neg][:n], cy_test[idx_neg][:n])), 'vis/ana_aneg.png')

    iutils.postprocess_images()


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
    #         argv = ['-d', ds, '-a', atk]
    #         print('\n$> ', argv)
    #         args = parser.parse_args(argv)
    #         detect(args)


