import numpy as np
import matplotlib.pyplot as plt
import keras
import imageio
from global_config import *

dataset = 'cxr'

clean = np.load('adversarial_medicine/numpy_to_share/%s/val_test_x.npy' % dataset).astype('float32')
label = np.load('adversarial_medicine/numpy_to_share/%s/val_test_y.npy' % dataset).argmax(axis=1)
keras.applications.inception_resnet_v2.preprocess_input(clean)
correct_idx, train_idx, test_idx = np.load('data/' + ADV_PREFIX + 'split_%s.npy' % dataset, allow_pickle=True)
clean = clean[correct_idx]
label = label[correct_idx]

#fgsm = np.load('data/' + ADV_PREFIX + 'Adv_%s_fgsm.npy' % dataset)[correct_idx]
#bim = np.load('data/' + ADV_PREFIX + 'Adv_%s_bim.npy' % dataset)[correct_idx]
pgd = np.load('data/' + ADV_PREFIX + 'Adv_%s_pgd.npy' % dataset)[correct_idx]
#cw = np.load('data/' + ADV_PREFIX + 'Adv_%s_cw-l2_0.npy' % dataset)[correct_idx]

def reg(x, scale=1): return ((x - x.min()) / (x.max() - x.min()) - 0.5) * scale + 0.5
plt.imshow(reg(clean[3]));plt.show()

idx = {'derm':1,  'dr':20,  'cxr':3}
imageio.imsave('vis/vis_%s_clean.png' % dataset, clean[idx[dataset]]/2+0.5)
imageio.imsave('vis/vis_%s_adv.png' % dataset, pgd[idx[dataset]]/2+0.5)
imageio.imsave('vis/vis_%s_diff.png' % dataset, reg(pgd[idx[dataset]]-clean[idx[dataset]], 30*2/255))

plt.figure(figsize=(16,12))
plt.imshow(np.concatenate([
    np.concatenate([
        reg(clean[0]),
        reg(fgsm[0]-clean[0]),
        reg(bim[0]-clean[0]),
        reg(pgd[0]-clean[0]),
        reg(cw[0] - clean[0])
    ], axis=1),
    np.concatenate([
        reg(clean[1]),
        reg(fgsm[1]-clean[1]),
        reg(bim[1]-clean[1]),
        reg(pgd[1]-clean[1]),
        reg(cw[1] - clean[1])
    ], axis=1),
    np.concatenate([
        reg(clean[2]),
        reg(fgsm[2]-clean[2]),
        reg(bim[2]-clean[2]),
        reg(pgd[2]-clean[2]),
        reg(cw[2] - clean[2])
    ], axis=1),
]))
plt.show()
