from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from models import get_model
from datasets import get_data
from callback_util import LoggerCallback, get_lr_scheduler
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from util import local_shuffle

def train(dataset='mnist', batch_size=128, epochs=50):
    """
    Train one model with data augmentation: random padding+cropping and horizontal flip
    :param args: 
    :return: 
    """
    print('Data set: %s, batch: %s, epochs: %s' % (dataset, batch_size, epochs))

    X_train, Y_train, X_test, Y_test = get_data(dataset, clip_min=-0.5, clip_max=0.5, onehot=True)

    n_images = X_train.shape[0]
    image_shape = X_train.shape[1:]
    n_class = Y_train.shape[1]
    print("n_images:", n_images, "n_class:", n_class, "image_shape:", image_shape)

    model = get_model(dataset, softmax=True)
    # model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # training with data augmentation
    if dataset == 'mnist':
        datagen = ImageDataGenerator(preprocessing_function=local_shuffle)
    else:
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            preprocessing_function=local_shuffle)
        datagen.fit(X_train)

    callbacks = []
    # acc, loss, lid
    log_callback = LoggerCallback(K.get_session(), model, X_test, Y_test, dataset, epochs)
    callbacks.append(log_callback)

    model_path = './model'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    cp_callback = ModelCheckpoint("model/shuffle_%s.hdf5" % dataset,
                                      monitor='acc',
                                      verbose=0,
                                      save_best_only=False,
                                      save_weights_only=True,
                                      period=10)
    callbacks.append(cp_callback)

    model.fit_generator(
        datagen.flow(X_train, Y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) / batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(X_test, Y_test),
        callbacks=callbacks)


def main(args):
    """
    Train model with data augmentation: random padding+cropping and horizontal flip
    :param args: 
    :return: 
    """
    assert args.dataset in ['mnist', 'cifar-10', 'svhn', 'all'], \
        "dataset parameter must be either 'mnist', 'cifar', 'svhn' or all"
    if args.dataset == 'all':
        for dataset in ['mnist', 'cifar-10', 'svhn']:
            train(dataset, args.batch_size, args.epochs)
    else:
        train(args.dataset, args.batch_size, args.epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar-10', 'svhn' or 'all'",
        required=True, type=str
    )
    parser.add_argument(
        '-e', '--epochs',
        help="The number of epochs to train for.",
        required=False, type=int
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False, type=int
    )
    parser.set_defaults(epochs=100)
    parser.set_defaults(batch_size=128)
    args = parser.parse_args()
    main(args)

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    #
    # args = parser.parse_args(['-d', 'cifar-10', '-e', '100', '-b', '128'])
    # main(args)

    K.clear_session()
