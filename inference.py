import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.applications.inception_resnet_v2 import preprocess_input

task = 'dr'

X = np.load('adversarial_medicine/numpy_to_share/{}/val_test_x.npy'.format(task))

model = keras.models.load_model('adversarial_medicine/model_to_share/{}/wb_model.h5'.format(task))

preprocess_input(X)
w = model.layers[-1].weights[0].eval(keras.backend.get_session())
w = np.abs(w[:, 1] - w[:, 0])
np.save('data/cweight_dr_wb.npy', (w, np.argsort(w)[::-1]))

feat_model = keras.models.Model(model.input, model.layers[-1].input)
feat = feat_model.predict(X, batch_size=200, verbose=1)
np.save('data/feat_dr_val.npy', feat)

# featdiff = np.mean(np.abs(feat_adv-feat), axis=0)
# plt.matshow(w[None, ...]);plt.show()
# plt.matshow(featdiff[None, ...]);plt.show()
pass

def reg(x): return (x - x.min()) / (x.max() - x.min())
plt.imshow(reg(X[0])); plt.show();