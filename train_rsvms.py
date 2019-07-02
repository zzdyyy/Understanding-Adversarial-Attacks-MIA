import random
import numpy as np
import sklearn.svm
from concurrent.futures import ThreadPoolExecutor
from sklearn.externals import joblib

important_ratio = 0.5
svm_dim = 30
svm_num = 10
svm_imp_ratio = 0.1

feat_train = np.load('data/feat_dr_train.npy')
Y = np.load('data/dr/train_y.npy')
y = Y[:, 1]

w, idx = np.load('data/cweight_dr_wb.npy')
idx = idx.astype(int)

imp_line = int(len(idx)*important_ratio)
imp_idx = idx[:imp_line]
unimp_idx = idx[imp_line:]

def train_svm(i):
    print('[svm %d] start training svm ...' % i)
    imp_num = int(svm_dim * svm_imp_ratio)
    f_idx = random.choices(imp_idx, k=imp_num) + random.choices(unimp_idx, k=svm_dim-imp_num)  # TODO: forget to re-balance
    X = feat_train[:, f_idx]
    X *= np.sum(w) / np.sum(w[f_idx])  # TODO: rescale is not saved
    svm = sklearn.svm.SVC(kernel='linear', probability=True)
    svm.fit(X, y)
    print('[svm %d] training over.' % i)
    print('[svm %d] accuracy score:' % i, sklearn.metrics.accuracy_score(y, svm.predict(X)))
    return f_idx, svm

with ThreadPoolExecutor(10) as executor:
    rsvms = executor.map(train_svm, range(svm_num))
rsvms = list(rsvms)

joblib.dump(rsvms, 'data/rsvms_dr.model')

feat = np.load('data/feat_dr_val.npy')
feat_adv = np.load('data/feat_dr_val_bim.npy')
is_adv = np.array([0] * len(feat) + [1] * len(feat_adv))

def svm_predict(args):
    f_idx, svm = args
    p_norm = svm.predict_proba(feat[:, f_idx])[:, 1:]
    p_adv = svm.predict_proba(feat_adv[:, f_idx])[:, 1:]
    return np.concatenate([p_norm, p_adv], axis=0)

with ThreadPoolExecutor(10) as executor:
    probs = executor.map(svm_predict, rsvms)
probs = list(probs)
probs = np.concatenate(probs, axis=1)

