import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.applications.inception_resnet_v2 import preprocess_input

task = 'dr'

X = np.load('adversarial_medicine/numpy_to_share/{}/val_test_x.npy'.format(task))
Y = np.load('adversarial_medicine/numpy_to_share/{}/val_test_y.npy'.format(task))

wb = keras.models.load_model('adversarial_medicine/model_to_share/{}/wb_model.h5'.format(task))
bb = keras.models.clone_model(wb)
bb.load_weights('adversarial_medicine/model_to_share/{}/bb_weights.hdf5'.format(task))
bb.compile(wb.optimizer, wb.loss, wb.metrics)

preprocess_input(X)
print(wb.evaluate(X, Y))
print(bb.evaluate(X, Y))
