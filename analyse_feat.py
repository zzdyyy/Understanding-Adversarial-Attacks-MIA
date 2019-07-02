import numpy as np
import matplotlib.pyplot as plt


w, idx = np.load('data/cweight_dr_wb.npy')
idx = idx.astype(int)

feat_val = np.load('data/feat_dr_val.npy')
feat_adv = np.load('data/feat_dr_val_fgsm.npy')

feat_diff = np.mean(np.abs(feat_adv-feat_val), axis=0)
idx2 = np.argsort(feat_diff)[::-1]

plt.plot(feat_diff[idx]);plt.show()

plt.plot(w[idx2]);plt.show()

plt.scatter(w, feat_diff); plt.show()