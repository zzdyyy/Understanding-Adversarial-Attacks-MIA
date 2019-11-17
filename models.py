"""
Date: 9/08/2018

Author: Xingjun Ma
Project: elastic_adv_defense
"""
from __future__ import absolute_import
from __future__ import print_function
import keras.backend as K
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Conv2D, Conv2DTranspose, Dense, MaxPooling2D, Dropout, \
    Flatten, Activation, BatchNormalization, Activation, AvgPool2D
from keras.models import Model
from keras.regularizers import l2

NUM_CLASS = {'mnist': 10, 'svhn': 10, 'cifar-10': 10, 'cifar-100': 100, 'dr':2, 'cxr':2, 'derm':2}


def get_model(dataset='cifar-10', softmax=True):
    """
    These models are those used in Madry's and Samuel G. Finlayson's paper.
    """
    if dataset == 'imagenet':
        model = keras.applications.resnet50.ResNet50(include_top=True)
        def predict_classes(x, batch_size=32, verbose=0):
            return model.predict(x, batch_size=batch_size, verbose=verbose).argmax(axis=-1)
        model.predict_classes = predict_classes  # add a useful function
        return model

    if dataset in ['dr', 'cxr', 'derm', 'cxr056', 'cxr0456', 'cxr05']:
        model = keras.models.load_model("model/model_%s.h5" % dataset)

        if not softmax:  # if don't need softmax activation
            old_model = model
            old_softmax = old_model.layers[-1]
            new_dense = Dense(old_softmax.output.shape[-1],
                              kernel_initializer=keras.initializers.Constant(old_softmax.weights[0].eval(K.get_session())),
                              bias_initializer=keras.initializers.Constant(old_softmax.weights[1].eval(K.get_session())),
                              name='dense_nosoftmax')
            x = old_softmax.input
            x = new_dense(x)

            model = Model(old_model.input, x)
            model.compile(optimizer=keras.optimizers.SGD(momentum=0.9),
                          loss='categorical_crossentropy', metrics=['accuracy'])

        def predict_classes(x, batch_size=32, verbose=0):
            return model.predict(x, batch_size=batch_size, verbose=verbose).argmax(axis=-1)
        model.predict_classes = predict_classes  # add a useful function

        return model

    if dataset == 'mnist':
        # MNIST model: 0, 2, 7, 10
        layers = [
            Conv2D(32, (3, 3), padding='valid', input_shape=(28, 28, 1)),
            BatchNormalization(),
            Activation('relu'),

            Conv2D(32, (3, 3)),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),

            Flatten(),

            Dense(128),
            BatchNormalization(),
            Activation('relu'),

            Dense(10),
        ]
    elif dataset == 'cifar-10':
        # CIFAR-10 model
        layers = [
            Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(32, (3, 3), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(64, (3, 3), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(64, (3, 3), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(128, (3, 3), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(128, (3, 3), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),

            Flatten(),

            Dense(256),
            BatchNormalization(),
            Activation('relu'),

            Dense(10),
        ]
    elif dataset == 'svhn':
        # SVHN model
        layers = [
            Conv2D(32, (3, 3), padding='valid', input_shape=(32, 32, 3)),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(64, (3, 3)),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(64, (3, 3)),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),

            Flatten(),

            Dense(256),
            BatchNormalization(),
            Activation('relu'),

            Dense(10),
        ]
    else:
        print("Add new type of model here such as cifar-100.")
        return

    model = Sequential()
    for layer in layers:
        model.add(layer)
    if softmax:
        model.add(Activation('softmax'))

    return model
